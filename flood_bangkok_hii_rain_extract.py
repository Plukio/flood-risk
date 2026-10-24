"""Fetch quality-screened daily HII rain for two Bangkok study districts.

The HII station timestamps are used as published; their local time zone was
not independently verified. Daily totals require every hourly reading.

Run: .venv/bin/python flood_bangkok_hii_rain_extract.py
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

from flood_bangkok_transfer import CACHE
from flood_bangkapi_surface_model import write_csv
from flood_road_data_pilot import HII_CATALOG, MISSING_RAIN, cached_get


STATIONS = ("BKK008", "BKK021", "AIT001", "BKC007",
            "BKK005", "BKK003", "CPY015", "BKK004")
OUT = CACHE / "hii_rain_daily_2022_2025.csv"
AUDIT = CACHE / "hii_rain_daily_2022_2025_audit.json"
YEAR_MONTHS = [(year, month) for year in range(2022, 2026)
               for month in range(1, 13)]


def one(job):
    station, year, month = job
    ym = f"{year}{month:02d}"
    raw = cached_get(CACHE / "hii_rain" / f"{ym}_{station}.csv",
                     f"{HII_CATALOG}/{year}/{ym}/{station}.csv", allow_404=True)
    return job, raw


def main() -> None:
    jobs = [(station, year, month) for station in STATIONS
            for year, month in YEAR_MONTHS]
    result = []
    audit = {"stations": list(STATIONS), "source_catalog": HII_CATALOG,
             "station_metadata": "https://data.hii.or.th/dataset/40afcc66-a44c-4f64-8242-ab8d555b342d/resource/daeae663-3aea-4995-8a45-8f114e189a31/download/0all_stn_metadata.csv",
             "missing_month_files": [], "suspect_all_zero_wet_months": [],
             "invalid_rows": 0, "duplicate_hours": 0, "conflicting_hours": 0,
             "complete_valid_station_days": 0, "station_day_coverage": {}}
    all_jobs = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        for future in as_completed([pool.submit(one, job) for job in jobs]):
            job, raw = future.result()
            all_jobs[job] = raw
    for station in STATIONS:
        covered = 0
        for year, month in YEAR_MONTHS:
            raw = all_jobs[(station, year, month)]
            if raw is None:
                audit["missing_month_files"].append(f"{station}:{year}{month:02d}")
                continue
            by_hour = {}
            for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
                stamp = row.get("measure_datetime") or (
                    (row.get("date") or "") + " " + (row.get("time") or "") + ":00")
                try:
                    when = datetime.fromisoformat(stamp)
                    if when.year != year or when.month != month:
                        raise ValueError("wrong month")
                except ValueError:
                    audit["invalid_rows"] += 1
                    continue
                value = row.get("rainfall_1h", row.get("rain", ""))
                try:
                    mm = float(value)
                    if mm < 0 or mm in MISSING_RAIN or mm > 200:
                        mm = None
                except (TypeError, ValueError):
                    mm = None
                key = when.replace(minute=0, second=0, microsecond=0)
                if key in by_hour:
                    audit["duplicate_hours"] += 1
                    if by_hour[key] is not None and mm is not None and by_hour[key] != mm:
                        audit["conflicting_hours"] += 1
                        by_hour[key] = None
                    elif by_hour[key] is None:
                        by_hour[key] = mm
                else:
                    by_hour[key] = mm
            valid = [mm for mm in by_hour.values() if mm is not None]
            suspect = (5 <= month <= 11 and len(valid) >= 480
                       and all(mm == 0 for mm in valid))
            if suspect:
                audit["suspect_all_zero_wet_months"].append(f"{station}:{year}{month:02d}")
            first = date(year, month, 1)
            last = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
            for offset in range((last - first).days + 1):
                day = first + timedelta(days=offset)
                hours = [by_hour.get(datetime.combine(day, datetime.min.time()) +
                                     timedelta(hours=h)) for h in range(24)]
                complete = not suspect and all(mm is not None for mm in hours)
                if complete:
                    covered += 1
                result.append({"station_code": station, "date": str(day),
                               "complete": int(complete),
                               "hours_valid": sum(mm is not None for mm in hours),
                               "rain_mm": round(sum(hours), 3) if complete else "",
                               "peak_1h_mm": round(max(hours), 3) if complete else "",
                               "qc": "all_zero_wet_month" if suspect else ""})
        audit["station_day_coverage"][station] = {
            "complete_days": covered, "days": 1461,
            "fraction": round(covered / 1461, 3)}
        print(station, audit["station_day_coverage"][station], flush=True)
    audit["complete_valid_station_days"] = sum(
        item["complete_days"] for item in audit["station_day_coverage"].values())
    audit["quality_rule"] = ("24 valid hourly readings, 0–200 mm/h, no conflicting duplicate, "
                             "exclude complete all-zero May–November months")
    audit["timestamp_limit"] = "HII timestamps are used as published; local time zone not independently verified"
    write_csv(OUT, result)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
