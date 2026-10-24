"""Build disjoint province patches for the fixed Bangkok transfer test.

Selection uses only 2026 GISTDA bbox counts in three dates, before inspecting
the Bangkok model result. Actual clipped polygon area is checked on all dates.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from flood_gistda_multiregion_data import (
    CACHE, CENTERS, GRID_M, OUTPUT, SIZE, TO_LONLAT, TO_UTM,
    labels, rain_features, static_features,
)
from flood_gistda_multiregion_worldcover import (
    CHANNELS, PIXEL_M, STAC, SUBSIZE, class_fractions,
)


ROOT = Path(__file__).resolve().parent
SCAN = OUTPUT / "province_candidate_scan_2026.json"
AUDIT = OUTPUT / "province_expansion_data_audit.json"


def patch_spec(lon: float, lat: float) -> dict:
    cx, cy = TO_UTM.transform(lon, lat)
    west = round(cx / GRID_M) * GRID_M - SIZE * GRID_M / 2
    north = round(cy / GRID_M) * GRID_M + SIZE * GRID_M / 2
    east, south = west + SIZE * GRID_M, north - SIZE * GRID_M
    corners = [TO_LONLAT.transform(x, y) for x in (west, east)
               for y in (south, north)]
    return {"center_lonlat": [lon, lat],
            "utm_bounds": [west, south, east, north],
            "query_bbox": [min(x for x, _ in corners) - .003,
                           min(y for _, y in corners) - .003,
                           max(x for x, _ in corners) + .003,
                           max(y for _, y in corners) + .003]}


def selected_specs() -> dict:
    scan = json.loads(SCAN.read_text())
    selected = {name: patch_spec(*center)
                for name, center in scan["centers"].items()
                if sum(count >= 100 for count in
                       scan["bbox_matched_counts"][name].values()) >= 2}
    if len(selected) != 15:
        raise ValueError(f"Expected 15 disjoint positive candidates, found {len(selected)}")
    bounds = {**{name: patch_spec(*center) for name, center in CENTERS.items()},
              **selected}
    names = list(bounds)
    for i, a in enumerate(names):
        ax0, ay0, ax1, ay1 = bounds[a]["utm_bounds"]
        for b in names[i+1:]:
            bx0, by0, bx1, by1 = bounds[b]["utm_bounds"]
            if min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0):
                raise ValueError(f"Province patches overlap: {a}, {b}")
    return selected


def worldcover(specs: dict) -> dict:
    bounds = [s["query_bbox"] for s in specs.values()]
    bbox = [min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds)]
    response = requests.post(STAC, json={
        "collections": ["esa-worldcover"], "bbox": bbox,
        "datetime": "2021-01-01T00:00:00Z/2021-12-31T23:59:59Z",
        "limit": 100}, timeout=45)
    response.raise_for_status()
    catalog = response.json()
    if any(link["rel"] == "next" for link in catalog.get("links", [])):
        raise ValueError("WorldCover STAC result paginated")
    items = catalog["features"]
    audit = {}
    for name, spec in specs.items():
        path = CACHE / f"{name}_worldcover_2021_250m.npz"
        if path.exists():
            with np.load(path) as saved:
                if (saved["values"].shape == (len(CHANNELS), SIZE, SIZE) and
                        tuple(saved["names"].astype(str)) == CHANNELS):
                    audit[name] = {"cached": True}
                    continue
        west, _, _, north = spec["utm_bounds"]
        transform = from_origin(west, north, PIXEL_M, PIXEL_M)
        land = np.zeros((SUBSIZE, SUBSIZE), np.uint8)
        tile_ids = []
        for item in items:
            ib = item["bbox"]
            pb = spec["query_bbox"]
            if ib[2] <= pb[0] or pb[2] <= ib[0] or ib[3] <= pb[1] or pb[3] <= ib[1]:
                continue
            url = planetary_computer.sign(item["assets"]["map"]["href"])
            with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
                with rasterio.open(url) as source:
                    with WarpedVRT(source, crs="EPSG:32647", transform=transform,
                                   width=SUBSIZE, height=SUBSIZE,
                                   resampling=Resampling.nearest) as warped:
                        tile = warped.read(1)
            overlap = (land != 0) & (tile != 0)
            if np.any(overlap & (land != tile)):
                raise ValueError(f"WorldCover tiles disagree: {name}")
            land[tile != 0] = tile[tile != 0]
            tile_ids.append(item["id"])
        values, coverage = class_fractions(land)
        np.savez_compressed(path, values=values, names=np.asarray(CHANNELS))
        audit[name] = {"tile_ids": tile_ids,
                       "valid_pixel_fraction": round(coverage, 5),
                       "mean_crop_fraction": round(float(values[1].mean()), 5)}
        print("WorldCover", name, audit[name], flush=True)
    return audit


def main() -> None:
    specs = selected_specs()
    with np.load(CACHE / "nong_chok_wet_250m.npz") as saved:
        days = [date.fromisoformat(str(x)) for x in saved["dates"]]
    coverage = labels(specs, days)
    static = static_features(specs)
    rain = rain_features(specs, days)
    cover = worldcover(specs)
    audit = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "selection_rule": "At least 100 bbox-matched polygons on at least two of 27 Sep, 4 Oct, 20 Oct 2026; disjoint 20 km patches",
             "note": "Selection used positive coverage, not Bangkok model performance. Bbox counts are not actual positive clipped area.",
             "dates": [str(d) for d in days], "patches": specs,
             "positive_coverage": coverage, "static": static,
             "rain": rain, "worldcover": cover}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print("saved", AUDIT, flush=True)


if __name__ == "__main__":
    main()
