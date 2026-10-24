"""Build open-data road/terrain/rain/radar audits for two Bangkok flood events.

Run: .venv/bin/python flood_road_data_pilot.py --site soi10
     .venv/bin/python flood_road_data_pilot.py --site bangkapi
The radar layer is a *candidate change mask*, not verified surface-water truth.
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import osmium
import planetary_computer
import rasterio
import requests
from shapely import transform as geometry_transform
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import from_origin, rowcol
from rasterio.vrt import WarpedVRT
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import substring


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "flood_road_pilot"
OUTPUT = ROOT / "artifacts" / "flood_road_pilot"
RAIN_CACHE = ROOT / "data" / "flood_road_pilot"
PBF = ROOT / "data" / "thailand-250101.osm.pbf"
DTM_URL = "https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer/exportImage"
HII_CATALOG = "https://tiservice.hii.or.th/opendata/data_catalog/hourly_rain"
HII_META = "https://data.hii.or.th/dataset/40afcc66-a44c-4f64-8242-ab8d555b342d/resource/daeae663-3aea-4995-8a45-8f114e189a31/download/0all_stn_metadata.csv"
RTC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TO_UTM = Transformer.from_crs(4326, 32647, always_xy=True)
TO_LONLAT = Transformer.from_crs(32647, 4326, always_xy=True)

# BMA flood-risk-point inventory places Chaeng Watthana Soi 10 here. BMA
# recorded flooding 17:00–18:20 local time on 13 Sep 2025; Sentinel-1 passed
# at 18:29. This later point inventory is geolocation, not a training label.
SITES = {
    "soi10": {"label": "Chaeng Watthana Soi 10, Lak Si",
              "lon": 100.578930, "lat": 13.887480,
              "event_date": "2025-09-13", "reference_date": "2025-09-01",
              "stations": ("BKK021", "BKK002"), "event_hours": (15, 18)},
    "bangkapi": {"label": "Bang Kapi intersection, Bang Kapi",
                 "lon": 100.648336, "lat": 13.766525,
                 "event_date": "2025-10-07", "reference_date": "2025-09-25",
                 "stations": ("BKK008", "BKK021", "BKC008"), "event_hours": (17, 20)},
}
CENTER_LON, CENTER_LAT = SITES["soi10"]["lon"], SITES["soi10"]["lat"]
EVENT_DATE = SITES["soi10"]["event_date"]
REFERENCE_DATE = SITES["soi10"]["reference_date"]
STATION_CODES = SITES["soi10"]["stations"]
EVENT_HOURS = SITES["soi10"]["event_hours"]
SITE_LABEL = SITES["soi10"]["label"]
TILE_METRES = 1000
S1_PIXELS = 100
ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "living_street", "service",
                "motorway_link", "trunk_link", "primary_link", "secondary_link",
                "tertiary_link"}
MISSING_RAIN = {-999, 9999, 999999}


def cached_get(path: Path, url: str, params: dict | None = None,
               allow_404: bool = False) -> bytes | None:
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=90,
                                    headers={"User-Agent": "Bangkok open-data flood pilot"})
            if allow_404 and response.status_code == 404:
                return None
            response.raise_for_status()
            path.write_bytes(response.content)
            return response.content
        except requests.RequestException as exc:
            error = exc
            time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download {url}: {error}")


def pilot_bounds() -> tuple[float, float, float, float]:
    x, y = TO_UTM.transform(CENTER_LON, CENTER_LAT)
    # Align 1 m DTM and 10 m radar grids at round UTM coordinates.
    x, y = round(x / 10) * 10, round(y / 10) * 10
    return x - 500, y - 500, x + 500, y + 500


def get_dtm(bounds: tuple[float, float, float, float]) -> Path:
    path = CACHE / "dtm_1000m_1m.tif"
    raw = cached_get(path, DTM_URL, {
        "bbox": ",".join(str(v) for v in bounds), "bboxSR": 32647,
        "imageSR": 32647, "size": "1000,1000", "format": "tiff",
        "pixelType": "F32", "f": "image"})
    assert raw is not None
    with rasterio.open(path) as src:
        if src.crs.to_epsg() != 32647 or any(abs(v - 1) > 0.01 for v in src.res):
            raise ValueError("LiDAR export is not a 1 m EPSG:32647 tile")
        if src.width != 1000 or src.height != 1000:
            raise ValueError("LiDAR export has unexpected dimensions")
    return path


class PilotRoads(osmium.SimpleHandler):
    def __init__(self, bounds: tuple[float, float, float, float]):
        super().__init__()
        self.bounds = bounds
        self.clip = box(*bounds)
        self.rows: list[dict] = []
        self.lon_min, self.lat_min = TO_LONLAT.transform(bounds[0], bounds[1])
        self.lon_max, self.lat_max = TO_LONLAT.transform(bounds[2], bounds[3])

    def way(self, way) -> None:
        road_type = way.tags.get("highway")
        if road_type not in ROAD_CLASSES:
            return
        nodes = [(n.lon, n.lat) for n in way.nodes if n.location.valid()]
        if len(nodes) < 2:
            return
        lon, lat = zip(*nodes)
        if (max(lon) < self.lon_min - 0.002 or min(lon) > self.lon_max + 0.002
                or max(lat) < self.lat_min - 0.002 or min(lat) > self.lat_max + 0.002):
            return
        xs, ys = TO_UTM.transform(lon, lat)
        geometry = LineString(zip(xs, ys))
        if not geometry.intersects(self.clip):
            return
        clipped = geometry.intersection(self.clip)
        pieces = [clipped] if clipped.geom_type == "LineString" else [
            g for g in getattr(clipped, "geoms", ()) if g.geom_type == "LineString"]
        for part_index, piece in enumerate(pieces):
            if piece.length < 10:
                continue
            for segment_index in range(math.ceil(piece.length / 50)):
                part = substring(piece, segment_index * 50,
                                 min(piece.length, (segment_index + 1) * 50))
                if part.length < 10:
                    continue
                self.rows.append({
                    "segment_id": f"osm:{way.id}:{part_index}:{segment_index}",
                    "osm_way_id": way.id, "name": way.tags.get("name", ""),
                    "name_en": way.tags.get("name:en", ""),
                    "highway": road_type, "bridge": way.tags.get("bridge", "no"),
                    "tunnel": way.tags.get("tunnel", "no"),
                    "length_m": round(part.length, 1), "line": part})


def sample_valid(src, points: list[tuple[float, float]]) -> list[float]:
    values = [float(a[0]) for a in src.sample(points)]
    return [v for v in values if math.isfinite(v) and -20 < v < 60]


def road_elevation(src, line: LineString) -> dict:
    fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
    points = [line.interpolate(f * line.length) for f in fractions]
    on_road = sample_valid(src, [(p.x, p.y) for p in points])
    centre = points[2]
    before = line.interpolate(max(0, line.length / 2 - 5))
    after = line.interpolate(min(line.length, line.length / 2 + 5))
    dx, dy = after.x - before.x, after.y - before.y
    norm = math.hypot(dx, dy) or 1
    side_points = [(centre.x + s * dy / norm * d,
                    centre.y - s * dx / norm * d)
                   for d in (20, 50) for s in (-1, 1)]
    sides = sample_valid(src, side_points)
    median = float(np.median(on_road)) if len(on_road) >= 3 else None
    side_median = float(np.median(sides)) if len(sides) >= 2 else None
    return {"lidar_road_median_m": round(median, 3) if median is not None else None,
            "lidar_road_p10_m": round(float(np.percentile(on_road, 10)), 3) if on_road else None,
            "lidar_side_median_m": round(side_median, 3) if side_median is not None else None,
            "lidar_road_minus_side_m": round(median - side_median, 3)
            if median is not None and side_median is not None else None,
            "lidar_valid_road_samples": len(on_road)}


def radar_scene(date: str, bounds: tuple[float, float, float, float]) -> dict:
    left, bottom = TO_LONLAT.transform(bounds[0], bounds[1])
    right, top = TO_LONLAT.transform(bounds[2], bounds[3])
    path = CACHE / f"rtc_stac_{date}_2day.json"
    raw = cached_get(path, RTC_STAC, {
        "collections": "sentinel-1-rtc",
        "bbox": f"{left},{bottom},{right},{top}",
        "datetime": f"{date}/{(datetime.fromisoformat(date) + timedelta(days=1)).date().isoformat()}",
        "limit": 20})
    assert raw is not None
    features = json.loads(raw).get("features", [])
    matching = [f for f in features
                if f["properties"].get("sat:relative_orbit") == 172
                and f["properties"].get("sat:orbit_state") == "ascending"
                and f["properties"]["datetime"].startswith(date)]
    if len(matching) != 1:
        raise ValueError(f"Expected one same-orbit RTC scene on {date}; found {len(matching)}")
    return matching[0]


def radar_band(scene: dict, band: str, transform) -> np.ndarray:
    href = planetary_computer.sign(scene["assets"][band]["href"])
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs="EPSG:32647", transform=transform,
                           width=S1_PIXELS, height=S1_PIXELS,
                           resampling=Resampling.bilinear) as vrt:
                return vrt.read(1, masked=True).filled(np.nan)


def get_radar(bounds: tuple[float, float, float, float]) -> tuple[dict, dict]:
    transform = from_origin(bounds[0], bounds[3], 10, 10)
    wet = radar_scene(EVENT_DATE, bounds)
    dry = radar_scene(REFERENCE_DATE, bounds)
    layers = {}
    for band in ("vv", "vh"):
        for name, scene in (("event", wet), ("reference", dry)):
            a = radar_band(scene, band, transform)
            layers[f"{name}_{band}_db"] = 10 * np.log10(np.maximum(a, 1e-7))
    delta = layers["event_vv_db"] - layers["reference_vv_db"]
    # Screening cue only: SAR darkening can come from geometry, vehicles,
    # surface roughness, vegetation and speckle. It is not a flood mask.
    candidate = ((layers["event_vv_db"] < -15) & (delta < -3)
                 & np.isfinite(layers["event_vh_db"])
                 & np.isfinite(layers["reference_vv_db"]))
    layers["vv_change_db"] = delta
    layers["darkening_candidate"] = candidate.astype(np.uint8)
    path = OUTPUT / "sentinel1_rtc_change_10m.tif"
    with rasterio.open(path, "w", driver="GTiff", width=S1_PIXELS,
                       height=S1_PIXELS, count=2, dtype="float32",
                       crs="EPSG:32647", transform=transform, nodata=-9999,
                       compress="deflate") as dst:
        dst.write(np.where(np.isfinite(delta), delta, -9999).astype("float32"), 1)
        dst.write(candidate.astype("float32"), 2)
        dst.set_band_description(1, "VV backscatter change, event minus reference, dB")
        dst.set_band_description(2, "Radar darkening candidate; NOT verified flood")
    polygons = []
    for geom, value in shapes(candidate.astype(np.uint8), mask=candidate,
                              transform=transform):
        if int(value) != 1:
            continue
        polygon_utm = shape(geom)
        polygon_lonlat = geometry_transform(polygon_utm, TO_LONLAT.transform,
                                            interleaved=False)
        polygons.append({"type": "Feature", "geometry": mapping(polygon_lonlat),
                         "properties": {"area_m2": round(polygon_utm.area),
                                        "label": "Radar darkening candidate, not verified flood"}})
    (OUTPUT / "radar_darkening_candidates.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": polygons}),
        encoding="utf-8")
    return layers, {"event_id": wet["id"], "reference_id": dry["id"],
                    "event_utc": wet["properties"]["datetime"],
                    "reference_utc": dry["properties"]["datetime"],
                    "same_relative_orbit": 172,
                    "candidate_pixels": int(candidate.sum()),
                    "valid_pixels": int(np.isfinite(delta).sum()),
                    "mask_definition": "event VV < -15 dB and VV drop > 3 dB; screening only"}


def rain_archive() -> tuple[list[dict], dict]:
    raw = cached_get(RAIN_CACHE / "hii_station_metadata.csv", HII_META)
    assert raw is not None
    stations = {r["Station_Code"]: r for r in
                csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))}
    station_codes = STATION_CODES
    jobs = [(code, year, month) for code in station_codes
            for year in range(2021, 2026) for month in range(1, 13)]

    def one(job):
        code, year, month = job
        ym = f"{year}{month:02d}"
        raw = cached_get(RAIN_CACHE / "hii" / f"{ym}_{code}.csv",
                         f"{HII_CATALOG}/{year}/{ym}/{code}.csv", allow_404=True)
        return job, raw

    all_rows = []
    missing_months = []
    suspicious_months = []
    duplicate_hours_removed = 0
    invalid_rows_removed = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        for future in as_completed([pool.submit(one, job) for job in jobs]):
            (code, year, month), raw = future.result()
            if raw is None:
                missing_months.append(f"{code}:{year}{month:02d}")
                continue
            monthly_by_time = {}
            for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
                value = row.get("rainfall_1h", row.get("rain", ""))
                timestamp = row.get("measure_datetime") or (
                    row["date"] + " " + row["time"] + ":00")
                try:
                    parsed = datetime.fromisoformat(timestamp)
                    if parsed.year != year or parsed.month != month:
                        raise ValueError("outside expected month")
                except ValueError:
                    invalid_rows_removed += 1
                    continue
                try:
                    mm = float(value)
                    if mm < 0 or mm in MISSING_RAIN:
                        mm = None
                except (ValueError, TypeError):
                    mm = None
                record = {"station_code": code,
                          "time_local_unverified": timestamp,
                          "rainfall_1h_mm": mm,
                          "quality_flag": row.get("quality_flag", "")}
                if timestamp in monthly_by_time:
                    duplicate_hours_removed += 1
                    old = monthly_by_time[timestamp]
                    if (old["rainfall_1h_mm"] is not None and mm is not None
                            and old["rainfall_1h_mm"] != mm):
                        raise ValueError(f"Conflicting HII rain readings: {code} {timestamp}")
                    if old["rainfall_1h_mm"] is None and mm is not None:
                        monthly_by_time[timestamp] = record
                else:
                    monthly_by_time[timestamp] = record
            monthly_rows = list(monthly_by_time.values())
            valid = [r["rainfall_1h_mm"] for r in monthly_rows
                     if r["rainfall_1h_mm"] is not None]
            # A complete all-zero month in Bangkok's wet season is a likely
            # stuck gauge; retain raw values but exclude it as evidence of dry weather.
            suspicious = (month in range(5, 12) and len(valid) >= 480
                          and all(v == 0 for v in valid))
            if suspicious:
                suspicious_months.append(f"{code}:{year}{month:02d}")
            for row in monthly_rows:
                row["qc"] = "suspect_all_zero_wet_month" if suspicious else ""
            all_rows.extend(monthly_rows)
    all_rows.sort(key=lambda r: (r["station_code"], r["time_local_unverified"]))
    with (OUTPUT / "hii_hourly_rain_2021_2025.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    meta = []
    for code in station_codes:
        row = stations[code]
        lat, lon = float(row["Latitude"]), float(row["Longitude"])
        distance = math.hypot((lat - CENTER_LAT) * 111000,
                              (lon - CENTER_LON) * 108000) / 1000
        station_rows = [r for r in all_rows if r["station_code"] == code]
        meta.append({"station_code": code, "name": row["Station_Name"],
                     "latitude": lat, "longitude": lon,
                     "distance_to_pilot_km": round(distance, 2),
                     "hours_available": sum(r["rainfall_1h_mm"] is not None and not r["qc"]
                                            for r in station_rows),
                     "hours_total": len(station_rows),
                     "months_missing": sum(x.startswith(code + ":") for x in missing_months),
                     "months_suspect_all_zero": sum(x.startswith(code + ":")
                                                    for x in suspicious_months)})
    with (OUTPUT / "hii_stations.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(meta[0]))
        writer.writeheader(); writer.writerows(meta)
    event = {}
    for code in station_codes:
        values = [r["rainfall_1h_mm"] for r in all_rows
                  if r["station_code"] == code
                  and r["time_local_unverified"].startswith(EVENT_DATE)
                  and f"{EVENT_HOURS[0]:02d}:00:00" <= r["time_local_unverified"][11:]
                  <= f"{EVENT_HOURS[1]:02d}:00:00"
                  and r["rainfall_1h_mm"] is not None and not r["qc"]]
        event[code] = {"hours_present": len(values),
                       "sum_mm": round(sum(values), 2) if values else None}
    return all_rows, {"stations": meta, "missing_months": missing_months,
                      "suspicious_all_zero_wet_months": suspicious_months,
                      "duplicate_hours_removed": duplicate_hours_removed,
                      "invalid_rows_removed": invalid_rows_removed,
                      f"event_hours_{EVENT_HOURS[0]}_to_{EVENT_HOURS[1]}": event,
                      "timestamp_note": "Archive timestamp has no UTC offset; treated as local Bangkok time for event comparison, unverified"}


def main() -> None:
    import argparse
    global CACHE, OUTPUT, CENTER_LON, CENTER_LAT, EVENT_DATE, REFERENCE_DATE
    global STATION_CODES, EVENT_HOURS, SITE_LABEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=sorted(SITES), default="soi10")
    args = parser.parse_args()
    site = SITES[args.site]
    CENTER_LON, CENTER_LAT = site["lon"], site["lat"]
    EVENT_DATE, REFERENCE_DATE = site["event_date"], site["reference_date"]
    STATION_CODES, EVENT_HOURS, SITE_LABEL = site["stations"], site["event_hours"], site["label"]
    if args.site != "soi10":
        CACHE = CACHE / args.site
        OUTPUT = OUTPUT / args.site
    if not PBF.exists():
        raise FileNotFoundError(f"Required OSM extract is missing: {PBF}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bounds = pilot_bounds()
    dtm = get_dtm(bounds)
    print("Loaded 1 m LiDAR tile", flush=True)
    roads = PilotRoads(bounds)
    roads.apply_file(str(PBF), locations=True)
    print(f"Extracted {len(roads.rows)} road pieces from OSM", flush=True)
    layers, radar_audit = get_radar(bounds)
    print("Loaded matched Sentinel-1 RTC scenes", flush=True)
    rain_rows, rain_audit = rain_archive()
    print(f"Loaded {len(rain_rows)} station-hour rows", flush=True)
    with rasterio.open(dtm) as src:
        for row in roads.rows:
            line = row["line"]
            row.update(road_elevation(src, line))
            midpoint = line.interpolate(line.length / 2)
            r, c = rowcol(from_origin(bounds[0], bounds[3], 10, 10),
                          midpoint.x, midpoint.y)
            valid = 0 <= r < S1_PIXELS and 0 <= c < S1_PIXELS
            row["radar_vv_change_db_at_midpoint"] = round(float(
                layers["vv_change_db"][r, c]), 2) if valid and np.isfinite(
                    layers["vv_change_db"][r, c]) else None
            row["radar_darkening_candidate_at_midpoint"] = int(
                layers["darkening_candidate"][r, c]) if valid else None
            row["terrain_not_road_surface"] = int(
                row["bridge"] != "no" or row["tunnel"] != "no")
    roads.rows.sort(key=lambda row: row["segment_id"])
    features = []
    for row in roads.rows:
        line = row.pop("line")
        coords = [TO_LONLAT.transform(x, y) for x, y in line.coords]
        features.append({"type": "Feature", "geometry": mapping(LineString(coords)),
                         "properties": row})
    (OUTPUT / "roads_lidar_radar.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features},
                   ensure_ascii=False), encoding="utf-8")
    audit = {"pilot": SITE_LABEL,
             "event_date": EVENT_DATE, "reference_date": REFERENCE_DATE,
             "pilot_bounds_epsg32647": bounds, "road_segments": len(features),
             "osm_distinct_ways": len({f["properties"]["osm_way_id"] for f in features}),
             "osm_snapshot": "2025-01-01 Thailand OSM PBF (cached)",
             "lidar_resolution_m": 1,
             "lidar_segments_valid": sum(f["properties"]["lidar_road_median_m"] is not None
                                         for f in features),
             "radar": radar_audit, "rain": rain_audit,
             "radar_candidate_road_midpoints": sum(
                 f["properties"]["radar_darkening_candidate_at_midpoint"] == 1
                 for f in features),
             "radar_candidate_non_bridge_midpoints": sum(
                 f["properties"]["radar_darkening_candidate_at_midpoint"] == 1
                 and f["properties"]["terrain_not_road_surface"] == 0
                 for f in features),
             "interpretation": "Radar darkening is not verified wet road; station rainfall may miss localized storms"}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "rain"}, indent=2))
    print("Stations:", json.dumps(rain_audit["stations"], ensure_ascii=False))
    print("Event rain:", json.dumps(
        rain_audit[f"event_hours_{EVENT_HOURS[0]}_to_{EVENT_HOURS[1]}"]))


if __name__ == "__main__":
    main()
