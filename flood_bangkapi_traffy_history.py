"""Stream BMA Traffy archives for 2022-2024 into minimal Bang Kapi extracts.

The source CSVs total roughly 1.2 GB but are not retained. Only dated flood
categories with coordinates and monthly all-category counts inside Bang Kapi
are saved. Free text, addresses and photo URLs are discarded.

Run: .venv/bin/python flood_bangkapi_traffy_history.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime

import requests
from shapely.geometry import Point

from flood_bangkapi_surface_grid import GRID_M, TO_UTM, district_polygon, grid_spec
from flood_bangkapi_traffy_extract import FLOOD_TERMS, OUTPUT, get_source


ARCHIVES = {
    2022: "1U3vn0aSQUUi3bSVuEVysDqvuD-663DI5",
    2023: "1vG1eQQp2LZeGnqGQ_hqhEfvNGB0ZlDio",
    2024: "1tW1MBfm1DNiA5fxARbNFhMLhhfWIBqn2",
}
FIELDNAMES = ["ticket_hash", "timestamp", "lon", "lat", "district", "type", "photo_present"]


def main() -> None:
    district = district_polygon()
    grid = grid_spec(district)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    flood_path = OUTPUT / "traffy_flood_2022_2024_minimal.csv"
    activity_path = OUTPUT / "traffy_all_activity_2022_2024_cell_month.csv"
    partial = flood_path.with_suffix(".partial.csv")
    all_activity = Counter()
    audits = {}
    with partial.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        with requests.Session() as session:
            for year, file_id in ARCHIVES.items():
                counts = Counter()
                categories = Counter()
                months = Counter()
                with get_source(session, file_id) as response:
                    reader = csv.DictReader(line.decode("utf-8-sig")
                                            for line in response.iter_lines())
                    required = {"ticket_id", "type", "problemtype_tag", "coords",
                                "district", "timestamp"}
                    if not required.issubset(reader.fieldnames or []):
                        raise ValueError(f"{year} Traffy CSV columns changed")
                    for row in reader:
                        counts["source_rows"] += 1
                        if counts["source_rows"] % 100000 == 0:
                            print(f"{year}: read {counts['source_rows']:,} rows; kept "
                                  f"{counts['bang_kapi_flood_rows']:,} Bang Kapi flood rows",
                                  flush=True)
                        try:
                            timestamp = datetime.fromisoformat(row["timestamp"])
                        except (ValueError, TypeError):
                            counts["invalid_timestamp"] += 1
                            continue
                        if timestamp.year != year:
                            counts["other_year_rows"] += 1
                            continue
                        counts["year_rows"] += 1
                        if row.get("district") != "บางกะปิ":
                            continue
                        try:
                            lon, lat = (float(v) for v in row["coords"].split(","))
                        except (ValueError, AttributeError):
                            counts["invalid_coords"] += 1
                            continue
                        if not (100.2 <= lon <= 101 and 13.4 <= lat <= 14.2):
                            counts["outside_bangkok_bbox"] += 1
                            continue
                        x, y = TO_UTM.transform(lon, lat)
                        if not district.covers(Point(x, y)):
                            counts["outside_district_polygon"] += 1
                            continue
                        col = math.floor((x - grid["west"]) / GRID_M)
                        grid_row = math.floor((grid["north"] - y) / GRID_M)
                        if not (0 <= grid_row < grid["rows"] and 0 <= col < grid["cols"]):
                            raise ValueError("Point is in district but outside grid")
                        counts["bang_kapi_all_category_rows"] += 1
                        all_activity[(grid_row, col, timestamp.strftime("%Y-%m"))] += 1
                        category = (row.get("type") or "") + " " + (row.get("problemtype_tag") or "")
                        if not any(term in category for term in FLOOD_TERMS):
                            continue
                        ticket = row.get("ticket_id") or ""
                        writer.writerow({"ticket_hash": hashlib.sha256(ticket.encode()).hexdigest()[:16],
                                         "timestamp": timestamp.isoformat(timespec="seconds"),
                                         "lon": lon, "lat": lat, "district": row["district"],
                                         "type": row["type"],
                                         "photo_present": int(bool(row.get("photo")))})
                        counts["bang_kapi_flood_rows"] += 1
                        categories[row["type"]] += 1
                        months[timestamp.strftime("%Y-%m")] += 1
                audits[str(year)] = {**dict(counts), "flood_categories": dict(categories),
                                     "flood_months": dict(sorted(months.items())),
                                     "source_catalog": f"https://data.bangkok.go.th/dataset/traffy-fondue-{year + 543}"}
    partial.replace(flood_path)
    with activity_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["grid_row", "grid_col", "month", "all_complaints"])
        for (grid_row, col, month), count in sorted(all_activity.items()):
            writer.writerow([grid_row, col, month, count])
    audit = {"archives": audits,
             "retained_flood_rows": sum(item.get("bang_kapi_flood_rows", 0)
                                        for item in audits.values()),
             "all_category_cell_months": len(all_activity),
             "privacy": "No free text, address or image URLs retained; raw ticket IDs hashed.",
             "label_limit": "Positive complaints only; an unreported cell-day is not verified dry."}
    (OUTPUT / "traffy_history_2022_2024_audit.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
