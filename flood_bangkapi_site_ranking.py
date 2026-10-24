"""Audit BMA Bang Kapi flood reports and rank named road locations.

Run: .venv/bin/python flood_bangkapi_site_ranking.py

This is an exploratory *site report-count* ranking, not a road-piece flood
forecast. The 2026 location inventory postdates the held-out 2025 reports.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import osmium
import rasterio
import requests
from pyproj import Transformer
from scipy.stats import spearmanr
from shapely.geometry import LineString, Point


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "flood_bangkapi_site_ranking"
OUTPUT = ROOT / "artifacts" / "flood_bangkapi_site_ranking"
REPORT_CACHE = ROOT / "data" / "flood_bangkok_transfer"
OSM_PBF = ROOT / "data" / "thailand-250101.osm.pbf"
INVENTORY_URL = ("https://data.bangkok.go.th/dataset/25bc3f04-5707-4b60-a885-"
                 "5ad45aa38cd6/resource/837b7fb6-ce1f-4356-881e-"
                 "6bf14eeb075f/download/flood_risk26.csv")
DTM_URL = "https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer/exportImage"
TO_UTM = Transformer.from_crs(4326, 32647, always_xy=True)
ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "living_street", "service",
                "motorway_link", "trunk_link", "primary_link", "secondary_link",
                "tertiary_link"}

# Every coordinate is read from the published inventory. The two Ramkhamhaeng
# Soi 33 entries have the same place name; BMA report text cannot split them.
# Combining them avoids silently assigning the same report to two points.
SITE_NAMES = {
    "motorway": ("ใต้สะพานมอเตอร์เวย์ พระราม9",),
    "ram43": ("ช่วงตรงข้ามซอยรามคำแหง 43",),
    "ram53": ("ช่วงตรงข้าม ซอยรามคำแหง 53",),
    "ram33": ("ช่วง ซ.รามคำแหง 33", "ช่วงซอยรามคำแหง 33"),
    "ram20": ("ช่วงซอยรามคำแหง 20",),
    "huamak24": ("ถ.หัวหมาก ช่วง ซ.รามคำแหง 24 แยก 18",),
    "police": ("ช่วงซอยตรงข้ามสถานีตำรวจนครบาลลาดพร้าว",),
    "greensport": ("ถ.ศรีนครินทร์ หน้าบริษัทกรีนสปอร์ต",),
    "latphrao110": ("ช่วงลาดพร้าว 110 -112",),
    "ram24uturn": ("จุดกลับรถรามคำแหง24 หน้าหอพักบ้านเรือทอง",),
    "bangkapi": ("ช่วงแยกบางกะปิ",),
    "lamsali": ("ช่วงแยกลำสาลี",),
    "ram19_1": ("ช่วงซอยรามคำแหง 19/1",),
}
LABELS = {
    "motorway": "Rama 9 motorway underpass",
    "ram43": "Ramkhamhaeng Soi 43 opposite",
    "ram53": "Ramkhamhaeng Soi 53 opposite",
    "ram33": "Ramkhamhaeng Soi 33 (two inventory points combined)",
    "ram20": "Ramkhamhaeng Soi 20",
    "huamak24": "Hua Mak / Ramkhamhaeng 24 intersection 18",
    "police": "Opposite Lat Phrao police station",
    "greensport": "Srinakarin / Green Sport",
    "latphrao110": "Lat Phrao Soi 110–112",
    "ram24uturn": "Ramkhamhaeng 24 U-turn / Ban Ruea Thong",
    "bangkapi": "Bang Kapi intersection",
    "lamsali": "Lam Sali intersection",
    "ram19_1": "Ramkhamhaeng Soi 19/1",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_inventory() -> dict[str, dict]:
    source = REPORT_CACHE / "flood_risk26.csv"
    if not source.exists():
        response = requests.get(INVENTORY_URL, timeout=90)
        response.raise_for_status()
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(response.content)
    with source.open(encoding="utf-8-sig", newline="") as file:
        bangkapi = [row for row in csv.DictReader(file)
                    if row["DISTRICT_NAME"].strip() == "บางกะปิ"]
    assert len(bangkapi) == 14, f"Expected 14 inventory points, found {len(bangkapi)}"
    found = {}
    for site, names in SITE_NAMES.items():
        matches = [row for row in bangkapi if row["STATION_NAME"] in names]
        if len(matches) != len(names):
            raise ValueError(f"Inventory names changed at {site}: {matches}")
        found[site] = {"label": LABELS[site], "inventory_names": list(names),
                       "points": [(float(row["UTM_X"]), float(row["UTM_Y"]))
                                  for row in matches]}
    assert sum(len(x["points"]) for x in found.values()) == len(bangkapi)
    return found


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value or "").replace("บ.", "บริษัท").replace("ซ.", "ซอย")


def match_report(road: str, area: str) -> tuple[str | None, str]:
    """Ordered, conservative place aliases; broad corridor text stays unmatched."""
    road, area = normalize(road), normalize(area)
    if "รามคำแหง" in road and re.search(r"\d+-\d+", area):
        return None, "broad or ambiguous Ramkhamhaeng road range"
    if "กรีนสปอร์ต" in area:
        if "แยกลำสาลี" in area:
            return None, "ambiguous Green Sport–Lam Sali corridor"
        return "greensport", "Green Sport name"
    if "มอเตอร์เวย์" in area and "พระราม9" in area:
        return "motorway", "Rama 9 motorway underpass"
    if "บ้านเรือทอง" in area and "กลับรถ" in area:
        return "ram24uturn", "named U-turn and dormitory"
    if "แยกลำสาลี" in area:
        return "lamsali", "Lam Sali intersection"
    if "แยกบางกะปิ" in area:
        return "bangkapi", "Bang Kapi intersection"
    if "สถานีตำรวจนครบาลลาดพร้าว" in area:
        return "police", "police station"
    if "ลาดพร้าว110-112" in area or "ลาดพร้าว110-112" in road + area:
        return "latphrao110", "Lat Phrao Soi 110–112"
    if "24แยก18" in area and "หัวหมาก" in road:
        return "huamak24", "Hua Mak / Ramkhamhaeng 24 intersection 18"
    if "รามคำแหง" in road or "รามคำแหง" in area:
        for number, site in (("19/1", "ram19_1"), ("20", "ram20"),
                             ("33", "ram33"), ("43", "ram43"), ("53", "ram53")):
            if re.search(r"(?:ซอย)?(?:รามคำแหง)?" + re.escape(number) + r"(?!\d|/)", area):
                return site, f"Ramkhamhaeng Soi {number}"
    return None, "no exact site alias"


def get_reports() -> list[dict]:
    rows = []
    for year in range(2021, 2026):
        source = REPORT_CACHE / f"bma_flood_{year}.csv"
        if not source.exists():
            raise FileNotFoundError(f"Run flood_bangkok_transfer.py to cache {source}")
        with source.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row["district"].replace("เขต", "").strip() == "บางกะปิ":
                    day = datetime.strptime(row["frd_flood_date_begin"], "%m/%d/%Y").date()
                    site, rule = match_report(row["frd_road"], row["frd_area"])
                    rows.append({"year": year, "date": day.isoformat(),
                                 "report_id": row["frd_id"], "road": row["frd_road"],
                                 "area": row["frd_area"], "site": site or "",
                                 "match_rule": rule,
                                 "flood_height_cm": row["frd_flood_height"]})
    return rows


def get_tile(site: str, index: int, lon: float, lat: float) -> Path:
    path = CACHE / f"dtm_{site}_{index}_256m_1m.tif"
    if not path.exists():
        x, y = TO_UTM.transform(lon, lat)
        response = requests.get(DTM_URL, params={
            "bbox": f"{x-128},{y-128},{x+128},{y+128}", "bboxSR": 32647,
            "imageSR": 32647, "size": "256,256", "format": "tiff",
            "pixelType": "F32", "f": "image"}, timeout=90)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path


def terrain_point(site: str, index: int, lon: float, lat: float) -> dict:
    path = get_tile(site, index, lon, lat)
    with rasterio.open(path) as src:
        if src.crs.to_epsg() != 32647 or src.width != 256 or src.height != 256 \
                or abs(src.res[0] - 1) > 0.01 or abs(src.res[1] - 1) > 0.01:
            raise ValueError(f"Unexpected LiDAR tile at {site}: {path}")
        values = src.read(1).astype(float)
    valid = np.isfinite(values) & (values > -20) & (values < 60)
    if np.mean(valid) < 0.95:
        raise ValueError(f"LiDAR coverage too low at {site}")
    row, col = np.indices(values.shape)
    radius = np.hypot(col - 127.5, row - 127.5)
    core = values[valid & (radius <= 20)]
    ring = values[valid & (radius >= 40) & (radius <= 110)]
    if len(core) < 1000 or len(ring) < 20000:
        raise ValueError(f"Too few LiDAR pixels at {site}")
    core_median, ring_median = float(np.median(core)), float(np.median(ring))
    return {"core_elevation_m": core_median,
            "ring_elevation_m": ring_median,
            "relative_elevation_m": core_median - ring_median,
            "valid_fraction": float(np.mean(valid))}


def get_terrain(sites: dict[str, dict]) -> dict[str, dict]:
    jobs = [(site, i, lon, lat) for site, meta in sites.items()
            for i, (lon, lat) in enumerate(meta["points"])]
    by_site: dict[str, list[dict]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(terrain_point, *job): job for job in jobs}
        for future in as_completed(futures):
            site = futures[future][0]
            by_site[site].append(future.result())
    return {site: {name: round(float(np.mean([p[name] for p in points])), 4)
                   for name in points[0]}
            for site, points in by_site.items()}


class NearestRoads(osmium.SimpleHandler):
    def __init__(self, sites: dict[str, dict]):
        super().__init__()
        self.points = {site: Point(*TO_UTM.transform(
            float(np.mean([p[0] for p in meta["points"]])),
            float(np.mean([p[1] for p in meta["points"]]))))
            for site, meta in sites.items()}
        all_points = [p for meta in sites.values() for p in meta["points"]]
        self.bounds = (min(p[0] for p in all_points) - 0.005,
                       min(p[1] for p in all_points) - 0.005,
                       max(p[0] for p in all_points) + 0.005,
                       max(p[1] for p in all_points) + 0.005)
        self.nearest = {site: (float("inf"), "", "") for site in sites}

    def way(self, way) -> None:
        road_class = way.tags.get("highway")
        if road_class not in ROAD_CLASSES:
            return
        nodes = [(node.lon, node.lat) for node in way.nodes if node.location.valid()]
        if len(nodes) < 2:
            return
        left, bottom, right, top = self.bounds
        if (min(p[0] for p in nodes) > right or max(p[0] for p in nodes) < left
                or min(p[1] for p in nodes) > top or max(p[1] for p in nodes) < bottom):
            return
        xs, ys = TO_UTM.transform(*zip(*nodes))
        line = LineString(zip(xs, ys))
        for site, point in self.points.items():
            distance = line.distance(point)
            if distance < self.nearest[site][0]:
                self.nearest[site] = (distance, way.tags.get("name", ""), road_class)


def get_nearest_roads(sites: dict[str, dict]) -> dict[str, dict]:
    """Audits point-to-OSM-road offsets; it does not snap reports to segments."""
    path = CACHE / "osm_nearest_road.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved.get("source_size_bytes") == OSM_PBF.stat().st_size
                and set(saved["sites"]) == set(sites)):
            return saved["sites"]
    handler = NearestRoads(sites)
    handler.apply_file(str(OSM_PBF), locations=True)
    result = {site: {"osm_nearest_road_m": round(distance, 1),
                     "osm_nearest_road_name": name,
                     "osm_nearest_road_class": road_class}
              for site, (distance, name, road_class) in handler.nearest.items()}
    if any(not np.isfinite(row["osm_nearest_road_m"]) for row in result.values()):
        raise ValueError("An inventory point has no nearby OSM drivable road")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": OSM_PBF.name,
                                "source_size_bytes": OSM_PBF.stat().st_size,
                                "sites": result}, ensure_ascii=False, indent=2) + "\n")
    return result


def pairwise_accuracy(scores: list[float], observed: list[int]) -> dict:
    correct = 0.0
    pairs = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if observed[i] == observed[j]:
                continue
            pairs += 1
            difference = (scores[i] - scores[j]) * (observed[i] - observed[j])
            correct += 1 if difference > 0 else 0.5 if difference == 0 else 0
    return {"comparable_pairs": pairs, "pairwise_agreement": round(correct / pairs, 4) if pairs else None}


def forward_report_baseline(reports: list[dict], sites: dict[str, dict]) -> dict:
    """Rank only sites known by 31 Aug, then check subsequent report days.

    The site cohort is built from earlier report text, so later reports do not
    select the evaluated sites. The 2026 inventory only provides map points.
    """
    cutoff = "2025-08-31"
    past: dict[str, set[str]] = defaultdict(set)
    future: dict[str, set[str]] = defaultdict(set)
    for report in reports:
        if not report["site"] or report["year"] != 2025:
            continue
        (past if report["date"] <= cutoff else future)[report["site"]].add(report["date"])
    known = set(past)
    rows = [{"site": site, "label": sites[site]["label"],
             "past_report_days_may_aug_2025": len(past[site]),
             "future_report_days_sep_dec_2025": len(future[site])}
            for site in known]
    rows.sort(key=lambda row: (-row["past_report_days_may_aug_2025"], row["site"]))
    for rank, row in enumerate(rows, 1):
        row["predicted_rank"] = rank
    write_csv(OUTPUT / "known_site_forward_ranking_2025.csv", rows)
    predicted = [row["past_report_days_may_aug_2025"] for row in rows]
    observed = [row["future_report_days_sep_dec_2025"] for row in rows]
    statistic = spearmanr(predicted, observed)
    return {"training_window": "2025-01-01 through 2025-08-31",
            "test_window": "2025-09-01 through 2025-12-31",
            "ranking_input": "BMA report days in the training window only",
            "known_site_count": len(known),
            "known_site_future_report_days": sum(observed),
            "all_future_report_days": sum(len(v) for v in future.values()),
            "newly_reported_sites": sorted(set(future) - known),
            "spearman": round(float(statistic.statistic), 4),
            "spearman_p_unadjusted": round(float(statistic.pvalue), 4),
            **pairwise_accuracy(predicted, observed)}


def main() -> None:
    sites = get_inventory()
    reports = get_reports()
    terrain = get_terrain(sites)
    osm_offsets = get_nearest_roads(sites)
    write_csv(OUTPUT / "report_match_audit.csv", reports)
    days = {year: defaultdict(set) for year in range(2021, 2026)}
    raw = {year: Counter() for year in range(2021, 2026)}
    for report in reports:
        if report["site"]:
            days[report["year"]][report["site"]].add(report["date"])
            raw[report["year"]][report["site"]] += 1

    rows = []
    for site, meta in sites.items():
        past = sum(len(days[year][site]) for year in range(2021, 2025))
        observed = len(days[2025][site])
        row = {"site": site, "label": meta["label"],
               "lon": round(float(np.mean([p[0] for p in meta["points"]])), 6),
               "lat": round(float(np.mean([p[1] for p in meta["points"]])), 6),
               "inventory_points": len(meta["points"]),
               "report_days_2021_2024": past,
               "report_rows_2025": raw[2025][site],
               "report_days_2025": observed,
               "report_dates_2025": ";".join(sorted(days[2025][site])),
               **terrain[site], **osm_offsets[site]}
        # The two terrain hypotheses are set by physical direction, not fit.
        row["low_elevation_score"] = -row["core_elevation_m"]
        row["local_depression_score"] = -row["relative_elevation_m"]
        row["historical_report_score"] = past
        rows.append(row)
    rows.sort(key=lambda row: (-row["report_days_2025"], row["site"]))
    for position, row in enumerate(rows, 1):
        row["observed_rank"] = position
    for name in ("low_elevation_score", "local_depression_score", "historical_report_score"):
        for rank, row in enumerate(sorted(rows, key=lambda r: (-r[name], r["site"])), 1):
            row[name.replace("score", "rank")] = rank
    write_csv(OUTPUT / "site_ranking_2025.csv", rows)

    observed = [row["report_days_2025"] for row in rows]
    metrics = {}
    for name in ("low_elevation_score", "local_depression_score", "historical_report_score"):
        predicted = [row[name] for row in rows]
        statistic = spearmanr(predicted, observed)
        metrics[name] = {"spearman": round(float(statistic.statistic), 4),
                         "spearman_p_unadjusted": round(float(statistic.pvalue), 4),
                         **pairwise_accuracy(predicted, observed)}
    audit = {"unit": "named location, with two Ramkhamhaeng Soi 33 inventory points combined",
             "test_year": 2025, "site_count": len(sites),
             "reports_by_year": {str(year): sum(r["year"] == year for r in reports)
                                 for year in range(2021, 2026)},
             "matched_reports_by_year": {str(year): sum(r["year"] == year and bool(r["site"])
                                            for r in reports) for year in range(2021, 2026)},
             "observed_2025_site_days": sum(observed),
             "distinct_2025_report_dates": len({r["date"] for r in reports if r["year"] == 2025}),
             "sites_more_than_20m_from_nearest_osm_road": sorted(
                 site for site, row in osm_offsets.items() if row["osm_nearest_road_m"] > 20),
             "metrics": metrics,
             "forward_report_baseline": forward_report_baseline(reports, sites),
             "caveat": "The site inventory was published in 2026 after the 2025 reports; this is an exploratory retrospective sample. Missing BMA reports do not prove dry roads."}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
