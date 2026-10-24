"""Stream public 2022–2024 Traffy reports for five, ten, or 49 districts.

Keeps only direct flood-category timestamp, coordinates, district, and type.
No comments, addresses, photos, or ticket identifiers are retained. The
source archives are streamed and discarded. Bang Kapi's older extract is
already provided by flood_bangkapi_traffy_history.py.

Run: .venv/bin/python flood_bangkok_traffy_history.py
     .venv/bin/python flood_bangkok_traffy_history.py --expanded
     .venv/bin/python flood_bangkok_traffy_history.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime

import requests
from shapely.geometry import Point

from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import TO_UTM
from flood_bangkapi_surface_model import DIRECT_TYPES
from flood_bangkapi_traffy_extract import get_source
from flood_bangkapi_traffy_history import ARCHIVES


CODES = ("1030", "1032", "1034", "1044", "1045")
MORE_CODES = ("1005", "1036", "1040", "1041", "1047")
OUT = CACHE / "traffy_flood_2022_2024_five_districts.csv"
AUDIT = CACHE / "traffy_flood_2022_2024_five_districts_audit.json"
FIELDS = ("timestamp", "lon", "lat", "district", "type")


def main(expanded: bool = False, all_districts: bool = False) -> None:
    if all_districts:
        codes = tuple(d["code"] for d in load_districts() if d["code"] != "1006")
    else:
        codes = CODES + MORE_CODES if expanded else CODES
    suffix = "49_districts" if all_districts else "ten_districts" if expanded else "five_districts"
    out = CACHE / f"traffy_flood_2022_2024_{suffix}.csv"
    audit_path = CACHE / f"traffy_flood_2022_2024_{suffix}_audit.json"
    districts = {d["name_th"]: d for d in load_districts() if d["code"] in codes}
    if len(districts) != len(codes):
        raise ValueError("Missing selected BMA district boundary")
    # The boundary DBF spells Rat Burana with ฏ; the Traffy archive uses ฎ.
    if "ราษฏร์บูรณะ" in districts:
        districts["ราษฎร์บูรณะ"] = districts["ราษฏร์บูรณะ"]
    CACHE.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.csv")
    audits = {}
    with partial.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        with requests.Session() as session:
            for year, file_id in ARCHIVES.items():
                counts = Counter()
                by_district = Counter()
                by_type = Counter()
                with get_source(session, file_id) as response:
                    reader = csv.DictReader(line.decode("utf-8-sig")
                                            for line in response.iter_lines())
                    required = {"type", "coords", "district", "timestamp"}
                    if not required.issubset(reader.fieldnames or []):
                        raise ValueError(f"{year} Traffy CSV columns changed")
                    for row in reader:
                        counts["source_rows"] += 1
                        if counts["source_rows"] % 100000 == 0:
                            print(f"{year}: read {counts['source_rows']:,} rows; "
                                  f"kept {counts['retained_flood_rows']:,}", flush=True)
                        district = districts.get(row.get("district"))
                        if district is None or row.get("type") not in DIRECT_TYPES:
                            continue
                        try:
                            when = datetime.fromisoformat(row["timestamp"])
                            lon, lat = (float(v) for v in row["coords"].split(","))
                        except (ValueError, TypeError, AttributeError):
                            counts["invalid_date_or_coords"] += 1
                            continue
                        if when.year != year or not (100.2 <= lon <= 101 and 13.4 <= lat <= 14.2):
                            counts["wrong_year_or_bbox"] += 1
                            continue
                        x, y = TO_UTM.transform(lon, lat)
                        if not district["polygon"].covers(Point(x, y)):
                            counts["outside_district_polygon"] += 1
                            continue
                        writer.writerow({"timestamp": when.isoformat(timespec="seconds"),
                                         "lon": lon, "lat": lat,
                                         "district": district["name_th"], "type": row["type"]})
                        counts["retained_flood_rows"] += 1
                        by_district[district["code"]] += 1
                        by_type[row["type"]] += 1
                audits[str(year)] = {**dict(counts), "by_district": dict(by_district),
                                     "by_type": dict(by_type),
                                     "source_catalog": f"https://data.bangkok.go.th/dataset/traffy-fondue-{year + 543}"}
    partial.replace(out)
    audit = {"district_codes": list(codes), "years": audits,
             "retained_flood_rows": sum(v.get("retained_flood_rows", 0) for v in audits.values()),
             "privacy": "Only timestamp, coordinates, district and direct flood category retained.",
             "label_limit": "A report is not a surveyed flood extent; missing reports are not verified dry."}
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", action="store_true",
                        help="Include five more districts: Bang Khen, Don Mueang, Bang Khae, Lak Si, Bang Na")
    parser.add_argument("--all", action="store_true", dest="all_districts",
                        help="Include all 49 districts other than Bang Kapi")
    args = parser.parse_args()
    if args.expanded and args.all_districts:
        parser.error("Choose only one of --expanded and --all")
    main(args.expanded, args.all_districts)
