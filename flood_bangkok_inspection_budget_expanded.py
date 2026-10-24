"""Eleven-district, three-season pre-October fixed-budget complaint backtest.

Run: .venv/bin/python flood_bangkok_inspection_budget_expanded.py
No Q4 outcome or weather from a test year enters that year's ranking.
"""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
from shapely.geometry import Point

from flood_bangkok_inspection_budget_backtest import (
    BUDGETS, cell_centres, model_rank, tie_expected,
)
from flood_bangkok_river_level_backtest import river_features, river_levels
from flood_bangkok_spatial_cnn_crossdistrict import DISTRICTS, report_labels, static_grid
from flood_bangkok_spatial_cnn_multiyear import report_rows
from flood_bangkok_spatial_water_state import load_district
from flood_bangkok_traffy_history import MORE_CODES
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import OUTPUT, TO_UTM, grid_spec, pump_points
from flood_bangkapi_surface_model import rainfall_by_date, write_csv


ARTIFACT = "surface_inspection_budget_expanded"
CODES = DISTRICTS + MORE_CODES
YEARS = (2023, 2024, 2025)


def cell_cache(district: dict, reports: dict, days: list[date],
               stations: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build the same 100 m report-buffer cell labels as district_data, compactly."""
    code = district["code"]
    path = OUTPUT / f"surface_shared_state_{code}_daily_cells.npz"
    if not path.exists():
        spec, static, _, cells = static_grid(district, stations)
        addresses = list(cells)
        cell_index = {address: i for i, address in enumerate(addresses)}
        rr = np.asarray([r for r, _ in addresses], dtype=np.int32)
        cc = np.asarray([c for _, c in addresses], dtype=np.int32)
        by_day = {day: i for i, day in enumerate(days)}
        observed = np.zeros((len(days), len(addresses)), dtype=np.uint8)
        for year in range(2022, 2026):
            labels, audit = report_labels(
                district, spec, cells,
                reports.get(district["name_th"], {}).get(year, []), year)
            for address, positive_days in labels.items():
                cell_i = cell_index[address]
                for day in positive_days:
                    observed[by_day[day], cell_i] = 1
            print("cache", code, year, audit, flush=True)
        np.savez_compressed(path, observed=observed,
                            static7=static[:, rr, cc].T, rows=rr, cols=cc)
    with np.load(path) as saved:
        rr, cc = saved["rows"].copy(), saved["cols"].copy()
        if saved["observed"].shape != (len(days), len(rr)):
            raise ValueError(f"Bad daily-cell cache: {code}")
    return rr, cc


def future_points(district: dict, reports: dict, year: int,
                  rr: np.ndarray, cc: np.ndarray) -> tuple[np.ndarray, dict]:
    spec = grid_spec(district["polygon"])
    positions = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rr, cc))}
    outcome = np.zeros(len(rr), dtype=np.int32)
    seen = set()
    outside = unmatched = 0
    for row in reports.get(district["name_th"], {}).get(year, []):
        day = date.fromisoformat(row["timestamp"][:10])
        if not (date(year, 10, 1) <= day <= date(year, 12, 31)):
            continue
        key = row["timestamp"], row["lon"], row["lat"]
        if key in seen:
            continue
        seen.add(key)
        x, y = TO_UTM.transform(float(row["lon"]), float(row["lat"]))
        if not district["polygon"].covers(Point(x, y)):
            outside += 1
            continue
        r = math.floor((spec["north"] - y) / 250)
        c = math.floor((x - spec["west"]) / 250)
        i = positions.get((r, c))
        if i is None:
            unmatched += 1
            continue
        outcome[i] += 1
    return outcome, {"unique_direct_category_records": len(seen),
                     "outside_district_polygon": outside,
                     "cell_unmatched": unmatched,
                     "mapped_report_points": int(outcome.sum()),
                     "affected_cells": int(np.count_nonzero(outcome))}


def case(code: str, year: int, district: dict, info: dict,
         rr: np.ndarray, cc: np.ndarray, reports: dict,
         days: list[date]) -> tuple[dict, list[dict]]:
    points, audit = future_points(district, reports, year, rr, cc)
    sites = (points > 0).astype(np.int32)
    cutoff_i = days.index(date(year, 9, 30)) + 1
    lookback_i = days.index(date(year - 1, 10, 1))
    recent = info["y"][lookback_i:cutoff_i].sum(axis=0).astype(np.float32)
    model_score, scenarios = model_rank(info, info, days, year, "baseline")
    rankings = {"local_baseline": model_score,
                "recent_report_days": recent,
                "lowest_lidar_ground": -info["static"][:, 0].astype(np.float32)}
    metrics = {}
    for budget in BUDGETS:
        if budget > len(rr):
            continue
        metrics[str(budget)] = {}
        for name, score in rankings.items():
            primary = tie_expected(score, sites, budget)
            secondary = tie_expected(score, points, budget)
            metrics[str(budget)][name] = {
                "captured_site_cells": round(primary, 3),
                "captured_report_points": round(secondary, 3),
                "site_cell_recall": round(primary / sites.sum(), 4) if sites.sum() else None,
                "report_point_recall": round(secondary / points.sum(), 4) if points.sum() else None,
            }
        metrics[str(budget)]["random_cells_expectation"] = {
            "captured_site_cells": round(budget / len(rr) * float(sites.sum()), 3),
            "captured_report_points": round(budget / len(rr) * float(points.sum()), 3),
            "site_cell_recall": round(budget / len(rr), 4),
            "report_point_recall": round(budget / len(rr), 4),
        }
    lon, lat = cell_centres(code, rr, cc)
    ranks = {}
    for name, score in rankings.items():
        order = np.lexsort((cc, rr, -score))
        ranks[name] = np.empty(len(rr), dtype=np.int32)
        ranks[name][order] = np.arange(1, len(rr) + 1)
    rows = []
    for i, (r, c) in enumerate(zip(rr, cc)):
        row = {"district_code": code, "district_name": district["name_en"],
               "year": year, "cell_id": f"{code}:{int(r):02d}:{int(c):02d}",
               "lon": round(float(lon[i]), 7), "lat": round(float(lat[i]), 7),
               "future_site_cell_audit_only": int(sites[i]),
               "future_report_points_audit_only": int(points[i])}
        for name, score in rankings.items():
            row[f"{name}_rank"] = int(ranks[name][i])
            row[f"{name}_score"] = round(float(score[i]), 8)
        rows.append(row)
    return {"district_code": code, "district_name": district["name_en"],
            "year": year, "cells": len(rr), "site_cells": int(sites.sum()),
            "report_points": int(points.sum()), "outcome_audit": audit,
            "historical_q4_rain_scenario_days": scenarios,
            "decision_date": f"{year}-09-30",
            "evaluation_dates": [f"{year}-10-01", f"{year}-12-31"],
            "budgets": metrics}, rows


def main() -> None:
    districts = {d["code"]: d for d in load_districts() if d["code"] in CODES}
    if len(districts) != len(CODES):
        raise ValueError("District metadata missing")
    reports = report_rows(expanded=True)
    stations = pump_points()
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   CACHE / "rain_1006_2021_2025.json"))
    levels, _ = river_levels()
    river = river_features(days, levels)
    cases, all_rows = {}, []
    for code in CODES:
        district = districts[code]
        rr, cc = cell_cache(district, reports, days, stations)
        info = load_district(code, days, river)
        for year in YEARS:
            result, rows = case(code, year, district, info, rr, cc, reports, days)
            cases[f"{code}_{year}"] = result
            all_rows.extend(rows)
            print(code, year, result["site_cells"], result["budgets"]["25"],
                  flush=True)
    output = {
        "purpose": "Retrospective fixed-budget pre-season cell ranking across 11 districts and three Q4 seasons",
        "decision_rule": "Train each district model through September 30; freeze report history; score using only earlier Q4 rainy-day weather scenarios",
        "outcome": "Future direct-category geocoded Traffy flood complaint points within district and 250 m cell; not verified inundation or dry-cell labels",
        "training_target": "100 m complaint buffers touching 250 m cells on the report day",
        "tie_rule": "Capture is expected under random selection at a score tie; CSV rank uses row and column tie break",
        "retrospective_static_data_limit": "2025 OSM canals used in 2023 and 2024; dates of availability for infrastructure inventory and LiDAR not established",
        "actual_bma_queue_or_costs_available": False,
        "budgets_cells": list(BUDGETS), "cases": cases,
    }
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUTPUT / f"{ARTIFACT}_cell_ranks.csv", all_rows)


if __name__ == "__main__":
    main()
