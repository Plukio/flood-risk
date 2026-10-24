"""Backtest longer wetness and lagged non-flood reporting activity.

Uses the same district report-count target and chronological splits as the
rain-only and river-level pilots. Activity is updated from the *prior* month
in a sequential backtest; the current month's report count is never an input.

Run after flood_bangkok_reporting_activity.py:
    .venv/bin/python flood_bangkok_shared_state_backtest.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from flood_bangkok_local_rate_calibration import CODES, count_model, features
from flood_bangkok_river_level_backtest import river_features, river_levels, scores
from flood_bangkok_spatial_cnn_multiyear import CAL_END, TRAIN_END, district_data, report_rows
from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import OUTPUT, RAIN_FEATURES, rainfall_by_date, write_csv


ACTIVITY = TRANSFER_CACHE / "traffy_nonflood_district_month_2022_2025.csv"
ARTIFACT = "surface_shared_state_count_backtest"


def preceding_wetness(days: list[date], path) -> np.ndarray:
    raw = json.loads(path.read_text())
    if raw.get("timezone") != "Asia/Bangkok":
        raise ValueError("Rain timestamps are not Bangkok local time")
    hourly = defaultdict(list)
    for stamp, mm in zip(raw["hourly"]["time"], raw["hourly"]["precipitation"]):
        hourly[date.fromisoformat(stamp[:10])].append(float(mm) if mm is not None else math.nan)
    totals = {day: sum(values) for day, values in hourly.items()
              if len(values) == 24 and all(math.isfinite(value) for value in values)}
    out = []
    for day in days:
        four_to_seven = [totals.get(day - timedelta(days=lag)) for lag in range(4, 8)]
        eight_to_fourteen = [totals.get(day - timedelta(days=lag)) for lag in range(8, 15)]
        if any(value is None for value in four_to_seven + eight_to_fourteen):
            raise ValueError(f"Missing 14-day antecedent rain before {day}")
        out.append((math.log1p(sum(four_to_seven)), math.log1p(sum(eight_to_fourteen))))
    return np.asarray(out, dtype=np.float32)


def nonflood_activity() -> dict[tuple[str, str], int]:
    with ACTIVITY.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    result = {}
    for row in rows:
        key = row["district_code"], row["month"]
        if key in result:
            raise ValueError(f"Duplicate activity month: {key}")
        result[key] = int(row["nonflood_reports"])
    for code in CODES:
        for month in range(1, 13):
            if (code, f"2025-{month:02d}") not in result:
                raise ValueError(f"Missing 2025 activity month for {code}: {month}")
    return result


def activity_feature(code: str, days: list[date], counts: dict) -> np.ndarray:
    values = []
    for day in days:
        previous_month_end = day.replace(day=1) - timedelta(days=1)
        key = code, previous_month_end.strftime("%Y-%m")
        count = counts.get(key)
        values.append(math.log1p(count / previous_month_end.day)
                      if count is not None else math.nan)
    return np.asarray(values, dtype=np.float32)[:, None]


def model_for(extra_monotonic: list[int]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=120,
        max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=20.0,
        monotonic_cst=[0, 0] + [1] * len(RAIN_FEATURES) + extra_monotonic,
        early_stopping=False, random_state=7)


def main() -> None:
    levels, river_audit = river_levels()
    activity = nonflood_activity()
    districts = {d["code"]: d for d in load_districts() if d["code"] in CODES}
    reports, stations = report_rows(expanded=True), pump_points()
    results, daily_rows = {}, []
    for code in CODES:
        district = districts[code]
        spec = grid_spec(district["polygon"])
        grid_data, target, (rr, cc), _ = district_data(
            district, reports, stations, spec["rows"], spec["cols"])
        rain_path = TRANSFER_CACHE / f"rain_{code}_2021_2025.json"
        rain = rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31), rain_path)
        days = sorted(rain)
        y = target[:, 0, rr, cc].sum(axis=1)
        np.savez_compressed(
            OUTPUT / f"surface_shared_state_{code}_daily_cells.npz",
            observed=target[:, 0, rr, cc].astype(np.uint8),
            static7=grid_data[0, :7][:, rr, cc].T,
            rows=rr, cols=cc)
        del grid_data, target
        base = features(days, rain)
        wetness = preceding_wetness(days, rain_path)
        past_activity = activity_feature(code, days, activity)
        river = river_features(days, levels)
        variants = {
            "rain_only": (base, []),
            "rain_plus_4_14d_wetness": (np.column_stack((base, wetness)), [1, 1]),
            "rain_plus_prior_nonflood_activity": (np.column_stack((base, past_activity)), [0]),
            "rain_plus_wetness_activity": (
                np.column_stack((base, wetness, past_activity)), [1, 1, 0]),
            "rain_plus_wetness_activity_river": (
                np.column_stack((base, wetness, past_activity, river)), [1, 1, 0, 0, 0]),
        }
        results[code] = {}
        for year in (2023, 2024, 2025):
            train_end = TRAIN_END if year == 2025 else date(year, 7, 31)
            cal_end = CAL_END if year == 2025 else date(year, 9, 30)
            test_end = date(year, 12, 31)
            n_train = sum(day <= train_end for day in days)
            n_cal = sum(day <= cal_end for day in days)
            n_test = sum(day <= test_end for day in days)
            observed = y[n_cal:n_test]
            rainy = np.asarray([rain[day]["rain_today_mm"] >= 10
                                for day in days[n_cal:n_test]])
            results[code][str(year)] = {}
            for name, (x, constraints) in variants.items():
                model = count_model() if name == "rain_only" else model_for(constraints)
                model.fit(x[:n_train], y[:n_train])
                raw_cal = model.predict(x[n_train:n_cal])
                factor = float(y[n_train:n_cal].sum() / raw_cal.sum())
                prediction = model.predict(x[n_cal:n_test]) * factor
                results[code][str(year)][name] = {
                    **scores(observed, prediction, rainy),
                    "aug_sep_calibration_multiplier": round(factor, 4)}
                if year == 2025:
                    for i, day in enumerate(days[n_cal:n_test]):
                        daily_rows.append({"district_code": code, "date": str(day),
                                           "variant": name, "observed": int(observed[i]),
                                           "predicted": round(float(prediction[i]), 6),
                                           "rain_today_mm": round(float(rain[day]["rain_today_mm"]), 3),
                                           "rain_4_7d_mm": round(float(np.expm1(wetness[n_cal+i, 0])), 3),
                                           "rain_8_14d_mm": round(float(np.expm1(wetness[n_cal+i, 1])), 3),
                                           "prior_month_nonflood_reports_per_day": round(float(
                                               np.expm1(past_activity[n_cal+i, 0])), 3)})
            print(code, year, {name: (value["predicted"], value["daily_mae"])
                               for name, value in results[code][str(year)].items()}, flush=True)
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps({
        "target": "Daily report-positive 250 m cells touched by 100 m Traffy buffers",
        "not_target": "verified flooding or water depth",
        "split": "Fit through July; calibrate on August–September; evaluate October–December in each year",
        "wetness": "Open-Meteo district-point rain totals at lags 4–7 and 8–14 days; lags 1–3 already in base features",
        "reporting_activity": "Previous calendar month's non-flood Traffy reports per day, district assigned in source; updated sequentially during test; excludes direct flood category; missing archive months remain missing, not zero",
        "activity_month_coverage": {
            code: {str(year): [f"{year}-{month:02d}" for month in range(1, 13)
                                if (code, f"{year}-{month:02d}") in activity]
                   for year in range(2022, 2026)} for code in CODES},
        "activity_source": str(ACTIVITY.relative_to(TRANSFER_CACHE.parent.parent)),
        "river_source": river_audit["source_url"],
        "results": results,
        "limitation": "Exploratory repeatedly inspected seasons; full-day rain is retrospective; past activity is reporting propensity proxy and can reflect other trends"},
        ensure_ascii=False, indent=2) + "\n")
    write_csv(OUTPUT / f"{ARTIFACT}_2025_daily.csv", daily_rows)


if __name__ == "__main__":
    main()
