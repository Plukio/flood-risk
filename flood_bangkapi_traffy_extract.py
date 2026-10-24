"""Stream public 2025 Traffy Fondue CSV; retain minimal flood report fields.

Run: .venv/bin/python flood_bangkapi_traffy_extract.py
The 662 MiB source is streamed and not saved. Free-text comments, photo URLs,
addresses and raw ticket IDs are discarded. Coordinates are reported points,
not surveyed flood extents. This is a positive-only complaint source.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests
import urllib3
from shapely.geometry import Point

from flood_bangkapi_surface_grid import TO_UTM, district_polygon, grid_spec, GRID_M


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "flood_bangkapi_surface"
FILE_ID = "1kJDdzkwhykAfEBwL49PW5OrRp3gxsz_R"
DOWNLOAD = "https://drive.usercontent.google.com/download"
# Public BMA catalog: https://data.bangkok.go.th/dataset/traffy-fondue-2568
FLOOD_TERMS = ("น้ำท่วม", "น้ำขัง")


def get_source(session: requests.Session, file_id: str = FILE_ID) -> requests.Response:
    # This machine's proxy presents a self-signed certificate for Google
    # Drive. This request carries no credentials and downloads public data.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    landing = session.get(DOWNLOAD, params={"id": file_id, "export": "download"},
                          timeout=60, verify=False)
    landing.raise_for_status()
    match = re.search(r'name="uuid" value="([^"]+)"', landing.text)
    if not match:
        raise ValueError("Google Drive download confirmation form changed")
    response = session.get(DOWNLOAD, params={"id": file_id, "export": "download",
                                             "confirm": "t", "uuid": match.group(1)},
                           timeout=180, verify=False, stream=True)
    response.raise_for_status()
    if "text/html" in response.headers.get("Content-Type", ""):
        raise ValueError("Expected the public Traffy CSV, received HTML")
    return response


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "traffy_flood_2025_minimal.csv"
    tmp = path.with_suffix(".partial.csv")
    total, matched, geocoded = 0, 0, 0
    district_counts, type_counts, month_counts = Counter(), Counter(), Counter()
    polygon = district_polygon()
    grid = grid_spec(polygon)
    all_activity = Counter()
    with requests.Session() as session, get_source(session) as response:
        reader = csv.DictReader(line.decode("utf-8-sig") for line in response.iter_lines())
        required = {"ticket_id", "type", "problemtype_tag", "coords", "district", "timestamp"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Traffy CSV columns changed")
        with tmp.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["ticket_hash", "timestamp", "lon", "lat",
                                                      "district", "type", "photo_present"])
            writer.writeheader()
            for row in reader:
                total += 1
                if total % 100000 == 0:
                    print(f"Read {total:,} source rows; retained {matched:,} flood rows", flush=True)
                if row.get("district") == "บางกะปิ" and row.get("timestamp"):
                    try:
                        lon, lat = (float(v) for v in row["coords"].split(","))
                        x, y = TO_UTM.transform(lon, lat)
                        if polygon.covers(Point(x, y)) and row["timestamp"].startswith("2025-"):
                            col = math.floor((x - grid["west"]) / GRID_M)
                            grid_row = math.floor((grid["north"] - y) / GRID_M)
                            if 0 <= grid_row < grid["rows"] and 0 <= col < grid["cols"]:
                                all_activity[(grid_row, col, row["timestamp"][:7])] += 1
                    except (ValueError, AttributeError):
                        pass
                category = (row.get("type") or "") + " " + (row.get("problemtype_tag") or "")
                if not any(term in category for term in FLOOD_TERMS):
                    continue
                try:
                    timestamp = datetime.fromisoformat(row["timestamp"])
                except (ValueError, TypeError):
                    continue
                if timestamp.year != 2025:
                    continue
                matched += 1
                try:
                    lon, lat = (float(v) for v in row["coords"].split(","))
                except (ValueError, AttributeError):
                    continue
                if not (100.2 <= lon <= 101 and 13.4 <= lat <= 14.2):
                    continue
                geocoded += 1
                writer.writerow({"ticket_hash": hashlib.sha256(row["ticket_id"].encode()).hexdigest()[:16],
                                 "timestamp": timestamp.isoformat(timespec="seconds"),
                                 "lon": lon, "lat": lat,
                                 "district": row["district"], "type": row["type"],
                                 "photo_present": int(bool(row.get("photo")))})
                district_counts[row["district"]] += 1
                type_counts[row["type"]] += 1
                month_counts[timestamp.strftime("%Y-%m")] += 1
    tmp.replace(path)
    with (OUTPUT / "traffy_all_activity_2025_cell_month.csv").open(
            "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["grid_row", "grid_col", "month", "all_complaints"])
        for (grid_row, col, month), count in sorted(all_activity.items()):
            writer.writerow([grid_row, col, month, count])
    audit = {"source": "BMA Traffy Fondue 2025 public CSV",
             "source_catalog": "https://data.bangkok.go.th/dataset/traffy-fondue-2568",
             "source_rows": total, "flood_category_rows": matched,
             "geocoded_bangkok_flood_rows": geocoded,
             "bang_kapi_district_rows": district_counts["บางกะปิ"],
             "bang_kapi_all_category_activity_rows": sum(all_activity.values()),
             "categories": dict(type_counts), "monthly_counts": dict(sorted(month_counts.items())),
             "privacy": "Saved only hashed ID, category, reported timestamp, coordinates, district and photo presence; no comments, addresses or image URLs.",
             "label_limit": "Positive complaint only; complaint time can follow flood onset, coordinates may be approximate, and missing complaints are not dry observations."}
    (OUTPUT / "traffy_flood_2025_audit.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
