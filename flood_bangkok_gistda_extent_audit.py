"""Fetch a GISTDA daily flood product and clip its positive polygons to BMA.

Usage: .venv/bin/python flood_bangkok_gistda_extent_audit.py 2026-10-20

The publicly listed data.go.th example key is loaded at runtime and is never
written to the outputs. An absent polygon is not a verified dry observation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
import shapely
from pyproj import Transformer
from shapely.geometry import mapping, shape

from flood_bangkok_transfer import load_districts


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "flood_bangkapi_surface"
CATALOG = "https://data.go.th/api/3/action/package_show"
COLLECTIONS = "https://api-gateway.gistda.or.th/api/2.0/resources/stac/flood/collections"
BMA_BBOX = "100.30,13.55,100.95,13.98"


def public_example_key() -> str:
    response = requests.get(CATALOG, params={"id": "flood-disaster-data"}, timeout=25)
    response.raise_for_status()
    resources = response.json()["result"]["resources"]
    for resource in resources:
        candidates = parse_qs(urlparse(resource.get("url", "")).query).get("api_key", [])
        if candidates and candidates[0]:
            return candidates[0]
    raise RuntimeError("No public example API key in the data.go.th catalog")


def fetch_day(day: date, key: str) -> tuple[dict, dict]:
    collection = f"flood_r2_{day:%Y%m%d}"
    response = requests.get(f"{COLLECTIONS}/{collection}/items",
                            params={"api_key": key, "limit": 20}, timeout=35)
    response.raise_for_status()
    items = response.json().get("features", [])
    if len(items) != 1 or "data" not in items[0].get("assets", {}):
        raise ValueError(f"Expected one daily item with a data asset: {collection}")
    item = items[0]
    asset = requests.get(item["assets"]["data"]["href"],
                         params={"bbox": BMA_BBOX, "limit": 10000, "offset": 0},
                         timeout=90)
    asset.raise_for_status()
    collection_data = asset.json()
    if collection_data.get("numberMatched") != len(collection_data.get("features", [])):
        raise ValueError("The daily asset was truncated; paginate before using it")
    return item, collection_data


def run(day: date) -> dict:
    key = public_example_key()
    item, data = fetch_day(day, key)
    districts = load_districts()
    to_utm = Transformer.from_crs(4326, 32647, always_xy=True).transform
    to_lonlat = Transformer.from_crs(32647, 4326, always_xy=True).transform
    output_features = []
    unique_sources = set()
    counts = Counter()
    area_m2 = defaultdict(float)
    sources = Counter()
    for index, feature in enumerate(data["features"]):
        geometry = shapely.transform(shape(feature["geometry"]), to_utm,
                                     interleaved=False)
        for district in districts:
            if not geometry.intersects(district["polygon"]):
                continue
            clipped = geometry.intersection(district["polygon"])
            if clipped.area <= 1:
                continue
            unique_sources.add(index)
            counts[district["name_en"]] += 1
            area_m2[district["name_en"]] += clipped.area
            source_name = str(feature.get("properties", {}).get("file_name", ""))
            sources[source_name] += 1
            output_features.append({"type": "Feature", "geometry": mapping(
                shapely.transform(clipped, to_lonlat, interleaved=False)), "properties": {
                    "day": day.isoformat(), "source_feature_index": index,
                    "district_code": district["code"],
                    "district": district["name_en"],
                    "area_m2": round(clipped.area, 2),
                    "source_file_name": source_name}})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"gistda_daily_{day:%Y%m%d}_bma"
    geojson = OUTPUT / f"{stem}.geojson"
    geojson.write_text(json.dumps({"type": "FeatureCollection",
                                   "features": output_features},
                                  ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_catalog": "https://data.go.th/dataset/flood-disaster-data",
        "collection": f"flood_r2_{day:%Y%m%d}",
        "stac_item_id": item.get("id"),
        "stac_datetime": item.get("properties", {}).get("datetime"),
        "note": "STAC daily datetime may be a date placeholder; source acquisition time is not independently verified",
        "bbox_query": [100.30, 13.55, 100.95, 13.98],
        "bbox_matched_polygons": data.get("numberMatched"),
        "polygons_intersecting_bma": len(unique_sources),
        "clipped_district_polygons": len(output_features),
        "district_polygon_counts": dict(counts),
        "district_area_km2": {name: round(area / 1_000_000, 4)
                              for name, area in area_m2.items()},
        "source_file_names": dict(sources),
        "geojson": str(geojson),
        "label_limit": "Positive mapped-water polygons only; outside polygons is not observed dry and source validation is not established",
    }
    (OUTPUT / f"{stem}_audit.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", type=date.fromisoformat, help="YYYY-MM-DD")
    print(json.dumps(run(parser.parse_args().date), ensure_ascii=False, indent=2))
