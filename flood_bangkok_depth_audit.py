"""Audit measured BMA road-flood heights against the complaint backtest.

Run: .venv/bin/python flood_bangkok_depth_audit.py
Road-text locations have no event coordinates; only an explicitly approximate
Bang Kapi 2025 site comparison is attached to cell ranks.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from flood_bangkok_inspection_budget_expanded import CODES, YEARS
from flood_bangkok_transfer import CACHE, load_districts, write_csv
from flood_bangkapi_surface_model import OUTPUT


ROOT = Path(__file__).resolve().parent
SITE_AUDIT = ROOT / "artifacts/flood_bangkapi_site_ranking/report_match_audit.csv"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def depth_stats(events: list[dict]) -> dict:
    values = np.asarray([event["depth_cm"] for event in events
                         if event["depth_cm"] is not None], dtype=float)
    return {"reports": len(events), "with_depth": len(values),
            "median_depth_cm": float(np.median(values)) if len(values) else None,
            "max_depth_cm": float(values.max()) if len(values) else None,
            "reports_at_least_20cm": int((values >= 20).sum()),
            "reports_at_least_30cm": int((values >= 30).sum())}


def bma_events() -> list[dict]:
    districts = {d["name_th"]: d for d in load_districts() if d["code"] in CODES}
    events = []
    for year in YEARS:
        for row in rows(CACHE / f"bma_flood_{year}.csv"):
            district = districts.get(row["district"].replace("เขต", "").strip())
            if district is None:
                continue
            day = datetime.strptime(row["frd_flood_date_begin"], "%m/%d/%Y").date()
            if day.year != year or day.month < 10:
                continue
            raw = row["frd_flood_height"].strip()
            depth = float(raw) if raw else None
            if depth is not None and (not np.isfinite(depth) or depth < 0):
                raise ValueError(f"Invalid BMA depth: {year} report {row['frd_id']}")
            events.append({"district_code": district["code"],
                           "district_name": district["name_en"],
                           "year": year, "date": day.isoformat(),
                           "report_id": row["frd_id"], "road": row["frd_road"],
                           "area_text": row["frd_area"], "depth_cm": depth,
                           "reported_flood_duration": row["frd_flood_total"]})
    return events


def bang_kapi_sites() -> list[dict]:
    matched = defaultdict(list)
    for row in rows(SITE_AUDIT):
        if row["year"] == "2025" and "2025-10-01" <= row["date"] <= "2025-12-31" \
                and row["site"] and row["flood_height_cm"]:
            matched[(row["date"], row["site"])].append(float(row["flood_height_cm"]))
    anchors = {}
    for row in rows(OUTPUT / "approximate_bma_wet_site_days_2025.csv"):
        if "2025-10-01" <= row["date_bangkok"] <= "2025-12-31":
            anchors[row["site"]] = row
            if (row["date_bangkok"], row["site"]) not in matched:
                raise ValueError("BMA site-day has no matched depth")
    ranks = {row["cell_id"]: row for row in rows(
        OUTPUT / "surface_inspection_budget_expanded_cell_ranks.csv")
        if row["district_code"] == "1006" and row["year"] == "2025"}
    result = []
    for site, anchor in sorted(anchors.items()):
        cid = anchor["approximate_cell_id"].replace("bk:", "1006:")
        if cid not in ranks:
            raise ValueError(f"BMA site outside mapped cell: {site}")
        selected = ranks[cid]
        depths = [max(values) for (day, name), values in matched.items()
                  if name == site]
        result.append({"site": site, "site_label": anchor["site_label"],
                       "approximate_cell_id": cid,
                       "bma_q4_site_days": len(depths),
                       "mean_depth_cm": round(float(np.mean(depths)), 2),
                       "max_depth_cm": max(depths),
                       "local_model_rank": int(selected["local_baseline_rank"]),
                       "recent_report_rank": int(selected["recent_report_days_rank"]),
                       "local_model_top25": int(int(selected["local_baseline_rank"]) <= 25),
                       "recent_report_top25": int(int(selected["recent_report_days_rank"]) <= 25),
                       "location_limit": "2026 BMA inventory point, approximate 2025 cell"})
    return result


def main() -> None:
    events = bma_events()
    by_year = {str(year): depth_stats([x for x in events if x["year"] == year])
               for year in YEARS}
    by_district = {}
    for code in sorted(CODES):
        district_events = [x for x in events if x["district_code"] == code]
        by_district[code] = {"district_name": district_events[0]["district_name"]
                             if district_events else next(d["name_en"] for d in load_districts()
                                                          if d["code"] == code),
                             **depth_stats(district_events)}
    known = [x["depth_cm"] for x in events if x["depth_cm"] is not None]
    distribution = {str(value): known.count(value) for value in sorted(set(known))}
    sites = bang_kapi_sites()
    result = {
        "source": "https://data.bangkok.go.th/dataset/frd_dds",
        "scope": "BMA major-road flood reports in Q4 2023-2025 for the same 11 districts as the inspection ranking backtest",
        "unit": "centimetres of recorded road flood height, as used in BMA archive",
        "all": depth_stats(events), "by_year": by_year,
        "by_district": by_district, "depth_value_counts": distribution,
        "bang_kapi_2025_q4_approximate_sites": {
            "sites": len(sites), "site_days": sum(x["bma_q4_site_days"] for x in sites),
            "max_depth_cm": max(x["max_depth_cm"] for x in sites),
            "top25_model_sites": sum(x["local_model_top25"] for x in sites),
            "top25_recent_report_sites": sum(x["recent_report_top25"] for x in sites)},
        "limits": [
            "BMA archive covers selected major-road events, not every Traffy complaint or flooded cell",
            "BMA road and area text lacks event coordinates; depths cannot be assigned to the 33 district-year cell outcomes",
            "Bang Kapi approximate cells use a 2026 inventory published after the 2025 outcome",
            "Reported depths are concentrated at 10, 15, and 20 cm and should not be treated as precise continuous measurements",
        ],
    }
    write_csv(OUTPUT / "surface_bma_road_depth_q4_2023_2025.csv", events)
    write_csv(OUTPUT / "surface_bma_depth_bang_kapi_2025_sites.csv", sites)
    (OUTPUT / "surface_bma_depth_audit_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all": result["all"], "by_year": by_year,
                      "bang_kapi_sites": result["bang_kapi_2025_q4_approximate_sites"]},
                     indent=2))


if __name__ == "__main__":
    main()
