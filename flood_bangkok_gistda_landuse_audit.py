"""Audit GISTDA source polygon flags on positive 2026 Bangkok days."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import shapely
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union

from flood_bangkok_gistda_extent_audit import OUTPUT, fetch_day, public_example_key
from flood_bangkok_transfer import load_districts


FIELDS = ("rice_area", "length_road", "building", "population")


def main() -> None:
    scan = json.loads((OUTPUT / "gistda_2026_bma_daily_scan.json").read_text())
    days = [date.fromisoformat(row["day"]) for row in scan["rows"]
            if row["polygons_intersecting_bma"] > 0]
    key = public_example_key()
    bma = unary_union([district["polygon"] for district in load_districts()])
    to_utm = Transformer.from_crs(4326, 32647, always_xy=True).transform
    rows = []
    for day in days:
        _, data = fetch_day(day, key)
        total = 0.0
        count = 0
        positive_count = {field: 0 for field in FIELDS}
        positive_area = {field: 0.0 for field in FIELDS}
        for feature in data["features"]:
            geometry = shapely.transform(shape(feature["geometry"]), to_utm,
                                         interleaved=False)
            if not geometry.intersects(bma):
                continue
            area = geometry.intersection(bma).area
            if area <= 1:
                continue
            total += area
            count += 1
            for field in FIELDS:
                if float(feature["properties"].get(field) or 0) > 0:
                    positive_count[field] += 1
                    positive_area[field] += area
        row = {"day": day.isoformat(), "bma_polygons": count,
               "bma_polygon_area_km2": round(total / 1e6, 5),
               "area_fraction_of_polygons_with_positive_source_field": {
                   field: round(positive_area[field] / total, 4) for field in FIELDS},
               "polygon_counts_with_positive_source_field": positive_count}
        rows.append(row)
        print(day, row["area_fraction_of_polygons_with_positive_source_field"], flush=True)
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "interpretation": "Fields are provided on source polygons. Positive rice_area/length_road signals an overlap, not a pixel-exact land classification or verified road flooding.",
              "rows": rows}
    path = OUTPUT / "gistda_2026_bma_landuse_flags.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
