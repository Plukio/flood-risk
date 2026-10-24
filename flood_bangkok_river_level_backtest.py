"""Test whether antecedent Chao Phraya level helps district report counts.

Target: count of 250 m cells touched by direct Traffy flood-report buffers.
This is an explanatory hindcast, not a water-depth or real-time forecast.

Run: .venv/bin/python flood_bangkok_river_level_backtest.py
"""

from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from flood_bangkok_local_rate_calibration import CODES, count_model, features
from flood_bangkok_spatial_cnn_multiyear import CAL_END, TRAIN_END, district_data, report_rows
from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import OUTPUT, RAIN_FEATURES, rainfall_by_date, write_csv


ROOT = Path(__file__).resolve().parent
RIVER_CSV = ROOT / "data/flood_bangkok_transfer/river_rama_daily_2022_2026.csv"
SOURCE_URL = ("https://data.bangkok.go.th/dataset/acddf9a1-da51-46c6-8964-a61e9a500ae3/"
              "resource/f01c4ce7-a185-4b47-9e71-3af9bd95d27c/download/river_rama.csv")
ARTIFACT = "surface_river_level_count_backtest"


def river_levels() -> tuple[dict[date, float], dict]:
    with RIVER_CSV.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    values = {}
    missing = []
    for row in rows:
        day = date.fromisoformat(row["DATE"])
        if day in values:
            raise ValueError(f"Duplicate river date: {day}")
        try:
            values[day] = float(row["MAX_WATER_LEVEL"].replace(",", ""))
        except ValueError:
            values[day] = math.nan
            missing.append(str(day))
    if min(values) != date(2022, 1, 1) or max(values) < date(2025, 12, 31):
        raise ValueError("River series does not cover the backtest")
    return values, {"source_url": SOURCE_URL, "station": "Rama VIII Bridge, Chao Phraya",
                    "first_date": str(min(values)), "last_date": str(max(values)),
                    "rows": len(rows), "missing_level_dates": missing,
                    "time_rule": "For day d use only reported maxima through d-1; no within-day maxima."}


def river_features(days: list[date], levels: dict[date, float]) -> np.ndarray:
    features_out = []
    for day in days:
        yesterday = levels.get(day - timedelta(days=1), math.nan)
        last_three = [levels.get(day - timedelta(days=lag), math.nan)
                      for lag in (1, 2, 3)]
        mean_three = float(np.mean(last_three)) if all(math.isfinite(v) for v in last_three) else math.nan
        features_out.append((yesterday, mean_three))
    return np.asarray(features_out, dtype=np.float32)


def level_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=120,
        max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=20.0,
        monotonic_cst=[0, 0] + [1] * len(RAIN_FEATURES) + [0, 0],
        early_stopping=False, random_state=7)


def scores(y: np.ndarray, pred: np.ndarray, rainy: np.ndarray) -> dict:
    return {
        "observed": int(y.sum()), "predicted": round(float(pred.sum()), 1),
        "daily_mae": round(float(mean_absolute_error(y, pred)), 4),
        "daily_poisson_deviance": round(float(mean_poisson_deviance(y, pred)), 4),
        "rainy_days_ge_10mm": int(rainy.sum()),
        "rainy_day_mae": round(float(mean_absolute_error(y[rainy], pred[rainy])), 4)
        if rainy.any() else None,
    }


def main() -> None:
    levels, audit = river_levels()
    districts = {d["code"]: d for d in load_districts() if d["code"] in CODES}
    reports, stations = report_rows(expanded=True), pump_points()
    results = {}
    daily_rows = []
    for code in CODES:
        district = districts[code]
        spec = grid_spec(district["polygon"])
        _, target, (rr, cc), _ = district_data(
            district, reports, stations, spec["rows"], spec["cols"])
        rain = rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                TRANSFER_CACHE / f"rain_{code}_2021_2025.json")
        days = sorted(rain)
        x_base = features(days, rain)
        x_river = np.column_stack([x_base, river_features(days, levels)])
        y = target[:, 0, rr, cc].sum(axis=1)
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
            variant = {}
            for name, x, model in (("rain_only", x_base, count_model()),
                                   ("rain_plus_prior_river", x_river, level_model())):
                model.fit(x[:n_train], y[:n_train])
                raw_cal = model.predict(x[n_train:n_cal])
                factor = float(y[n_train:n_cal].sum() / raw_cal.sum())
                pred = model.predict(x[n_cal:n_test]) * factor
                variant[name] = {**scores(observed, pred, rainy),
                                 "aug_sep_calibration_multiplier": round(factor, 4)}
                if year == 2025:
                    for day, actual, p in zip(days[n_cal:n_test], observed, pred):
                        daily_rows.append({"district_code": code, "date": str(day),
                                           "variant": name, "observed": int(actual),
                                           "predicted": round(float(p), 6),
                                           "prior_day_river_max": x_river[days.index(day), -2],
                                           "prior_three_day_river_mean": x_river[days.index(day), -1]})
            results[code][str(year)] = variant
            print(code, year, variant, flush=True)
    audit["feature_names"] = ["prior_day_river_max", "prior_three_day_river_mean"]
    audit["lag_coverage_2022_2025"] = {
        "complete_prior_day": sum(math.isfinite(levels.get(date(2022, 1, 1) + timedelta(days=i-1), math.nan))
                                  for i in range(1461)),
        "days": 1461,
    }
    output = {"target": "daily district count of report-positive 250 m cells, not inundation",
              "model": "same monotone Poisson rain-count model, adding two unconstrained lagged river features",
              "rain_timing": "Rain features include full-day observed rain; this is a retrospective scenario model",
              "river_audit": audit, "results": results}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        output, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUTPUT / f"{ARTIFACT}_2025_daily.csv", daily_rows)


if __name__ == "__main__":
    main()
