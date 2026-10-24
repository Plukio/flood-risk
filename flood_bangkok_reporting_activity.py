"""Stream public Traffy archives into district-month non-flood report counts.

The raw archives, text, coordinates, and identifiers are not retained. These
counts can only be used with a lag, never for the current prediction month.

Run: .venv/bin/python flood_bangkok_reporting_activity.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime

import requests

from flood_bangkapi_traffy_extract import FILE_ID, FLOOD_TERMS, get_source
from flood_bangkapi_traffy_history import ARCHIVES
from flood_bangkok_transfer import CACHE, load_districts


OUTPUT = CACHE / "traffy_nonflood_district_month_2022_2025.csv"
AUDIT = CACHE / "traffy_nonflood_district_month_2022_2025_audit.json"
CODES = ("1006", "1040")


def main() -> None:
    names = {d["name_th"]: d["code"] for d in load_districts() if d["code"] in CODES}
    if set(names.values()) != set(CODES):
        raise ValueError("Missing requested district boundaries")
    monthly = Counter()
    audits = {}
    with requests.Session() as session:
        for year, file_id in {**ARCHIVES, 2025: FILE_ID}.items():
            counts = Counter()
            with get_source(session, file_id) as response:
                reader = csv.DictReader(line.decode("utf-8-sig") for line in response.iter_lines())
                required = {"timestamp", "district", "type", "problemtype_tag"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"{year} Traffy columns changed")
                for row in reader:
                    counts["source_rows"] += 1
                    code = names.get(row.get("district"))
                    if code is None:
                        continue
                    try:
                        when = datetime.fromisoformat(row["timestamp"])
                    except (ValueError, TypeError):
                        counts["invalid_timestamp"] += 1
                        continue
                    if when.year != year:
                        counts["wrong_year"] += 1
                        continue
                    category = (row.get("type") or "") + " " + (row.get("problemtype_tag") or "")
                    if any(term in category for term in FLOOD_TERMS):
                        counts[f"{code}_excluded_flood_category"] += 1
                        continue
                    monthly[(code, when.strftime("%Y-%m"))] += 1
                    counts[f"{code}_nonflood_reports"] += 1
                    if counts["source_rows"] % 100000 == 0:
                        print(year, f"{counts['source_rows']:,} rows", flush=True)
            audits[str(year)] = dict(counts)
            print(year, audits[str(year)], flush=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["district_code", "month", "nonflood_reports"])
        for (code, month), value in sorted(monthly.items()):
            writer.writerow([code, month, value])
    AUDIT.write_text(json.dumps({
        "source_catalogs": [f"https://data.bangkok.go.th/dataset/traffy-fondue-{y + 543}"
                            for y in (*ARCHIVES, 2025)],
        "districts": list(CODES), "years": audits, "retained_month_rows": len(monthly),
        "count_rule": "District assigned in source; valid source-year timestamp; exclude rows whose type or tag contains น้ำท่วม or น้ำขัง.",
        "privacy": "Only district-month counts are retained; no text, coordinates, images, or IDs."},
        ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
