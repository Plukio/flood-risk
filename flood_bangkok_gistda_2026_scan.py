"""Find 2026 GISTDA daily products with actual water polygons inside BMA.

Read the public example key at runtime. Never persist item/asset URLs, which
can include credentials. Only positive BMA days are clipped to GeoJSON.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone

import requests
import shapely
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union

from flood_bangkok_gistda_extent_audit import (
    COLLECTIONS, OUTPUT, fetch_day, public_example_key, run,
)
from flood_bangkok_transfer import load_districts


def check(day: date, key: str, bma_polygon) -> dict:
    try:
        item, collection = fetch_day(day, key)
    except ValueError as error:
        return {"day": day.isoformat(), "status": "no_usable_daily_item",
                "reason": str(error), "bbox_matched_polygons": None,
                "polygons_intersecting_bma": 0}
    to_utm = Transformer.from_crs(4326, 32647, always_xy=True).transform
    overlapping = 0
    for feature in collection["features"]:
        geometry = shapely.transform(shape(feature["geometry"]), to_utm,
                                     interleaved=False)
        if geometry.intersects(bma_polygon) and geometry.intersection(bma_polygon).area > 1:
            overlapping += 1
    return {"day": day.isoformat(), "status": "checked",
            "stac_item_id": item.get("id"),
            "bbox_matched_polygons": collection.get("numberMatched"),
            "polygons_intersecting_bma": overlapping}


def main() -> None:
    key = public_example_key()
    response = requests.get(COLLECTIONS, params={"api_key": key, "limit": 3000},
                            timeout=90)
    response.raise_for_status()
    catalog = response.json()
    if catalog.get("context", {}).get("returned") != catalog.get("context", {}).get("matched"):
        raise ValueError("Collection catalog truncated")
    days = sorted(date.fromisoformat(c["id"][-8:-4] + "-" + c["id"][-4:-2] + "-" + c["id"][-2:])
                  for c in catalog["collections"] if c["id"].startswith("flood_r2_2026"))
    bma = unary_union([district["polygon"] for district in load_districts()])
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(check, day, key, bma): day for day in days}
        for future in as_completed(tasks):
            result = future.result()
            rows.append(result)
            print(result["day"], result["bbox_matched_polygons"],
                  result["polygons_intersecting_bma"], flush=True)
    rows.sort(key=lambda row: row["day"])
    positive = [row for row in rows if row["polygons_intersecting_bma"]]
    for row in positive:
        row["clipped_audit"] = run(date.fromisoformat(row["day"]))["geojson"]
    audit = {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source": "https://data.go.th/dataset/flood-disaster-data",
             "note": "Actual asset polygons intersected with BMA; no-polygon cells are unknown, not dry",
             "days_checked": len(rows), "positive_days": len(positive),
             "rows": rows}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "gistda_2026_bma_daily_scan.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print("saved", path, "positive days", len(positive), flush=True)


if __name__ == "__main__":
    main()
