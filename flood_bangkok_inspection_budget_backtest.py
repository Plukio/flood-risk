"""Pre-season fixed-budget cell inspection backtest using open Bangkok data.

No test-period rain or reports enter a cell's rank. Future Traffy reports are
only a proxy for verified problem sites. BMA road-site checks are approximate.

Run: .venv/bin/python flood_bangkok_inspection_budget_backtest.py
"""

from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from flood_bangkok_new_feature_ablation import feature_matrix, load_group
from flood_bangkok_river_level_backtest import river_features, river_levels
from flood_bangkok_spatial_cnn_multiyear import report_rows
from flood_bangkok_spatial_water_state import load_district, model, training_indices
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import OUTPUT, TO_LONLAT, TO_UTM, grid_spec
from flood_bangkapi_surface_model import rainfall_by_date, write_csv


ROOT = Path(__file__).resolve().parent
ARTIFACT = "surface_inspection_budget_backtest"
BUDGETS = (10, 25, 50, 100)
MODEL_VARIANTS = ("baseline", "+surface_cover")
RAIN_SCENARIO_MIN_MM = 10.0


def cell_centres(code: str, rr: np.ndarray, cc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    district = next(d for d in load_districts() if d["code"] == code)
    spec = grid_spec(district["polygon"])
    x = spec["west"] + (cc + 0.5) * 250
    y = spec["north"] - (rr + 0.5) * 250
    return np.asarray(TO_LONLAT.transform(x, y))


def future_traffy(code: str, year: int, rr: np.ndarray,
                  cc: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    district = next(d for d in load_districts() if d["code"] == code)
    spec = grid_spec(district["polygon"])
    positions = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rr, cc))}
    reports = report_rows(expanded=True).get(district["name_th"], {}).get(year, [])
    report_count = np.zeros(len(rr), dtype=np.int32)
    report_days = [set() for _ in rr]
    seen, outside, unmatched = set(), 0, 0
    for row in reports:
        day = date.fromisoformat(row["timestamp"][:10])
        if day < date(year, 10, 1) or day > date(year, 12, 31):
            continue
        key = (row["timestamp"], row["lon"], row["lat"])
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
        report_count[i] += 1
        report_days[i].add(day)
    return (report_count, np.asarray([len(x) for x in report_days]),
            {"unique_report_points": len(seen), "outside_district": outside,
             "point_cell_unmatched": unmatched,
             "mapped_report_points": int(report_count.sum()),
             "affected_cells": int(np.count_nonzero(report_count))})


def bma_bang_kapi_q4(rr: np.ndarray, cc: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    path = OUTPUT / "approximate_bma_wet_site_days_2025.csv"
    positions = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rr, cc))}
    sites = [set() for _ in rr]
    site_days = np.zeros(len(rr), dtype=np.int32)
    outside = 0
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if not ("2025-10-01" <= row["date_bangkok"] <= "2025-12-31"):
                continue
            cell = row["approximate_cell_id"]
            if not cell:
                outside += 1
                continue
            _, r, c = cell.split(":")
            i = positions.get((int(r), int(c)))
            if i is None:
                outside += 1
                continue
            sites[i].add(row["site"])
            site_days[i] += 1
    site_count = np.asarray([len(x) for x in sites], dtype=np.int32)
    return site_count, site_days, {
        "bma_named_sites": int(site_count.sum()),
        "bma_site_days": int(site_days.sum()),
        "site_days_unmatched": outside,
        "location_limit": "BMA road-flood dates joined to 2026 inventory coordinates; approximate 2025 cells"}


def tie_expected(score: np.ndarray, outcome: np.ndarray, budget: int) -> float:
    """Expected captured outcomes if cells tied at the cutoff are randomised."""
    if budget >= len(score):
        return float(outcome.sum())
    cutoff = np.partition(score, len(score) - budget)[len(score) - budget]
    above = score > cutoff
    tied = score == cutoff
    remaining = budget - int(above.sum())
    return float(outcome[above].sum() + remaining / tied.sum() * outcome[tied].sum())


def model_rank(source: dict, target: dict, days: list[date], year: int,
               variant: str) -> tuple[np.ndarray, int]:
    cutoff = date(year, 9, 30)
    n_train = days.index(cutoff) + 1
    train_di, train_ci, weights = training_indices(source["y"], n_train)
    fitted = model().fit(feature_matrix(source, train_di, train_ci, variant),
                         source["y"][train_di, train_ci], sample_weight=weights)
    # Use only historical Q4 wet days known by the September decision date. Their
    # weather is a fixed portfolio of scenarios; no future rain is supplied.
    scenario_di = np.asarray([i for i, day in enumerate(days)
                              if day < date(year, 1, 1) and day.month >= 10
                              and math.expm1(float(target["dynamic"][i, 2]))
                              >= RAIN_SCENARIO_MIN_MM], dtype=np.int32)
    if len(scenario_di) < 5:
        raise ValueError(f"Too few historical rainy Q4 days for {year}")
    snapshot = target["prior"][days.index(cutoff + timedelta(days=1))]
    scoring = {**target, "prior": np.broadcast_to(snapshot, target["prior"].shape)}
    cell_count = target["cells"]
    di = np.repeat(scenario_di, cell_count)
    ci = np.tile(np.arange(cell_count), len(scenario_di))
    score = fitted.predict_proba(feature_matrix(scoring, di, ci, variant))[:, 1]
    return score.reshape(len(scenario_di), cell_count).mean(axis=0), len(scenario_di)


def one_case(code: str, year: int, data: dict, days: list[date],
             rows_cols: dict) -> tuple[dict, list[dict]]:
    target = data[code]
    rr, cc = rows_cols[code]
    if code == "1040":
        sites, days_outcome, outcome_audit = future_traffy(code, year, rr, cc)
        primary_name, secondary_name = "future_report_site_cells", "future_report_points"
        secondary = sites
        primary = (sites > 0).astype(np.int32)
    else:
        primary, secondary, outcome_audit = bma_bang_kapi_q4(rr, cc)
        primary_name, secondary_name = "bma_named_road_sites", "bma_road_site_days"
        days_outcome = secondary
    n_cutoff = days.index(date(year, 9, 30)) + 1
    lookback = days.index(date(year - 1, 10, 1))
    recent = target["y"][lookback:n_cutoff].sum(axis=0).astype(np.float32)
    ranking = {"recent_report_days": recent,
               "lowest_lidar_ground": -target["static"][:, 0].astype(np.float32)}
    scenario_days = {}
    for source_code, label in ((code, "local"),
                                ("1006" if code == "1040" else "1040", "transfer")):
        for variant in MODEL_VARIANTS:
            name = f"{label}_{variant.replace('+', 'plus_')}"
            ranking[name], scenario_days[name] = model_rank(
                data[source_code], target, days, year, variant)
    budgets = {}
    for k in BUDGETS:
        if k > len(rr):
            continue
        budgets[str(k)] = {name: {
            "captured_primary": round(tie_expected(score, primary, k), 3),
            "captured_secondary": round(tie_expected(score, secondary, k), 3),
            "primary_recall": round(tie_expected(score, primary, k) / primary.sum(), 4)
            if primary.sum() else None,
            "secondary_recall": round(tie_expected(score, secondary, k) / secondary.sum(), 4)
            if secondary.sum() else None,
        } for name, score in ranking.items()}
        budgets[str(k)]["random_cells_expectation"] = {
            "captured_primary": round(k / len(rr) * float(primary.sum()), 3),
            "captured_secondary": round(k / len(rr) * float(secondary.sum()), 3),
            "primary_recall": round(k / len(rr), 4),
            "secondary_recall": round(k / len(rr), 4)}
    local_at_25 = tie_expected(ranking["local_baseline"], primary, 25)
    history_to_match = next(k for k in range(1, len(rr) + 1)
                            if tie_expected(recent, primary, k) >= local_at_25)
    lon, lat = cell_centres(code, rr, cc)
    ranks = {}
    for name, score in ranking.items():
        order = np.lexsort((cc, rr, -score))
        ordinal = np.empty(len(rr), dtype=np.int32)
        ordinal[order] = np.arange(1, len(rr) + 1)
        ranks[name] = ordinal
    cells = []
    for i, (r, c, lo, la) in enumerate(zip(rr, cc, lon, lat)):
        row = {"district_code": code, "year": year,
               "cell_id": f"{code}:{int(r):02d}:{int(c):02d}",
               "row": int(r), "col": int(c),
               "lon": round(float(lo), 7), "lat": round(float(la), 7),
               "future_primary_outcome_for_audit_only": int(primary[i]),
               "future_secondary_outcome_for_audit_only": int(secondary[i]),
               "future_report_days_for_audit_only": int(days_outcome[i])}
        for name, score in ranking.items():
            row[f"{name}_rank"] = int(ranks[name][i])
            row[f"{name}_score"] = round(float(score[i]), 8)
        cells.append(row)
    return ({"district_code": code, "year": year,
             "decision_date": f"{year}-09-30",
             "evaluation_dates": [f"{year}-10-01", f"{year}-12-31"],
             "cells": len(rr), "primary_outcome": primary_name,
             "secondary_outcome": secondary_name,
             "primary_total": int(primary.sum()),
             "secondary_total": int(secondary.sum()),
             "outcome_audit": outcome_audit,
             "historical_q4_rain_scenario_days": scenario_days,
             "effort_proxy": {
                 "local_baseline_sites_at_25_cells": round(local_at_25, 3),
                 "recent_report_cells_to_match": history_to_match,
                 "model_cells_saved_vs_recent_report_heuristic": history_to_match - 25,
                 "sign_rule": "Positive favours the local model; negative favours recent reports",
                 "limit": "Retrospective complaint/approximate-road-site capture, not verified inspections or monetary savings"},
             "budgets": budgets}, cells)


def main() -> None:
    levels, _ = river_levels()
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   CACHE / "rain_1006_2021_2025.json"))
    river = river_features(days, levels)
    data, rows_cols = {}, {}
    for code in ("1006", "1040"):
        info = load_district(code, days, river)
        with np.load(OUTPUT / f"surface_shared_state_{code}_daily_cells.npz") as saved:
            rr, cc = saved["rows"], saved["cols"]
        info["new_static"], _ = load_group(code, "new_static", rr, cc)
        data[code], rows_cols[code] = info, (rr, cc)
    cases, cells = {}, []
    for code, year in (("1040", 2024), ("1040", 2025), ("1006", 2025)):
        result, rows = one_case(code, year, data, days, rows_cols)
        cases[f"{code}_{year}"] = result
        cells.extend(rows)
        print(code, year, "primary", result["primary_total"],
              "K25", result["budgets"]["25"], flush=True)
    result = {
        "purpose": "Retrospective fixed-budget pre-season cell selection; not a field inspection trial",
        "site_limit": "Bang Khae uses future geocoded Traffy flood complaints as a proxy, not subsequently verified wet sites; Bang Kapi BMA road sites are dated independent positive records with approximate 2026 anchors",
        "decision_rule": "Rank at 30 September using training through that date, prior-year October-December rain scenarios known by then, and report history observed by then; no test-quarter weather or reports enter ranks",
        "baseline_limit": "Recent report days and low ground are transparent heuristic comparators; BMA's actual inspection policy and cost data are unavailable",
        "tie_rule": "Capture is the expected result under random selection among cells tied at the budget cutoff; ranked CSV uses row/column as stable tie break for display",
        "budgets_cells": list(BUDGETS),
        "cases": cases,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        result, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUTPUT / f"{ARTIFACT}_cell_ranks.csv", cells)
    queue = sorted((row for row in cells if row["district_code"] == "1040"
                    and row["year"] == 2025 and row["local_baseline_rank"] <= 25),
                   key=lambda row: row["local_baseline_rank"])
    write_csv(OUTPUT / f"{ARTIFACT}_bang_khae_2025_historical_top25.csv", queue)


if __name__ == "__main__":
    main()
