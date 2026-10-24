"""Test local rain-conditioned daily complaint-volume calibration.

For Bang Kapi and Bang Khae, fit a small monotone Poisson tree to each
district's 2022–July 2025 positive cell-day counts using rain and season.
Use Aug–Sep 2025 only for a multiplicative level correction, and evaluate
Oct–Dec 2025. This is a count model for the report proxy, not water depth.

Run: .venv/bin/python flood_bangkok_local_rate_calibration.py
"""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkok_spatial_cnn_multiyear import CAL_END, TRAIN_END, district_data, report_rows
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import OUTPUT, RAIN_FEATURES, rainfall_by_date, write_csv


CODES = ("1006", "1040")
ARTIFACT = "surface_local_rain_count_calibration_2025"


def count_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=120,
        max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=20.0,
        monotonic_cst=[0, 0] + [1] * len(RAIN_FEATURES),
        early_stopping=False, random_state=7)


def features(days: list[date], rain: dict) -> np.ndarray:
    return np.asarray([[math.sin(2 * math.pi * day.month / 12),
                        math.cos(2 * math.pi * day.month / 12),
                        *(math.log1p(max(0, rain[day][key])) for key in RAIN_FEATURES)]
                       for day in days], dtype=np.float32)


def main() -> None:
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
        X = features(days, rain)
        y = target[:, 0, rr, cc].sum(axis=1)
        train_count = sum(day <= TRAIN_END for day in days)
        cal_end = sum(day <= CAL_END for day in days)
        model = count_model()
        model.fit(X[:train_count], y[:train_count])
        raw_cal = model.predict(X[train_count:cal_end])
        raw_test = model.predict(X[cal_end:])
        cal_ratio = float(y[train_count:cal_end].sum() / raw_cal.sum())
        adjusted = raw_test * cal_ratio
        recent_rate = float(y[train_count:cal_end].mean())
        observed = y[cal_end:]
        for day, actual, raw, corrected in zip(days[cal_end:], observed, raw_test, adjusted):
            daily_rows.append({"district_code": code, "date": str(day),
                               "observed_positive_cell_days": int(actual),
                               "raw_predicted_positive_cell_days": round(float(raw), 6),
                               "calibrated_predicted_positive_cell_days": round(float(corrected), 6),
                               "recent_rate_predicted_positive_cell_days": round(recent_rate, 6)})
        test_months = {}
        for month in sorted({day.strftime("%Y-%m") for day in days[cal_end:]}):
            selected = np.asarray([day.strftime("%Y-%m") == month for day in days[cal_end:]])
            test_months[month] = {"observed_positive_cell_days": int(observed[selected].sum()),
                                  "raw_count_model": round(float(raw_test[selected].sum()), 1),
                                  "calibrated_count_model": round(float(adjusted[selected].sum()), 1)}
        results[code] = {"district": district["name_en"], "cells": len(rr),
                         "train_positive_cell_days": int(y[:train_count].sum()),
                         "calibration_positive_cell_days": int(y[train_count:cal_end].sum()),
                         "calibration_raw_predicted_positive_cell_days": round(float(raw_cal.sum()), 1),
                         "calibration_multiplier": round(cal_ratio, 4),
                         "test_positive_cell_days": int(observed.sum()),
                         "test_raw_predicted_positive_cell_days": round(float(raw_test.sum()), 1),
                         "test_calibrated_predicted_positive_cell_days": round(float(adjusted.sum()), 1),
                         "test_recent_calibration_rate_baseline": round(recent_rate * len(observed), 1),
                         "test_raw_daily_count_mae": round(float(mean_absolute_error(observed, raw_test)), 4),
                         "test_calibrated_daily_count_mae": round(float(mean_absolute_error(observed, adjusted)), 4),
                         "test_recent_rate_daily_count_mae": round(float(mean_absolute_error(
                             observed, np.full(len(observed), recent_rate))), 4),
                         "test_months": test_months}
        historical_backtests = {}
        for year in (2023, 2024):
            earlier_train = sum(day <= date(year, 7, 31) for day in days)
            earlier_cal = sum(day <= date(year, 9, 30) for day in days)
            earlier_test = sum(day <= date(year, 12, 31) for day in days)
            hindcast = count_model().fit(X[:earlier_train], y[:earlier_train])
            cal_prediction = hindcast.predict(X[earlier_train:earlier_cal])
            test_prediction = hindcast.predict(X[earlier_cal:earlier_test])
            adjustment = float(y[earlier_train:earlier_cal].sum() /
                               cal_prediction.sum())
            predicted = test_prediction * adjustment
            actual = y[earlier_cal:earlier_test]
            recent = float(y[earlier_train:earlier_cal].mean())
            historical_backtests[str(year)] = {
                "train_dates": ["2022-01-01", f"{year}-07-31"],
                "calibration_dates": [f"{year}-08-01", f"{year}-09-30"],
                "test_dates": [f"{year}-10-01", f"{year}-12-31"],
                "observed_positive_cell_days": int(actual.sum()),
                "calibrated_predicted_positive_cell_days": round(float(predicted.sum()), 1),
                "recent_rate_predicted_positive_cell_days": round(recent * len(actual), 1),
                "calibrated_daily_count_mae": round(float(mean_absolute_error(
                    actual, predicted)), 4),
                "recent_rate_daily_count_mae": round(float(mean_absolute_error(
                    actual, np.full(len(actual), recent))), 4)}
        results[code]["historical_backtests"] = historical_backtests
        print(code, results[code], flush=True)
    metrics = {"target": "district count of 250 m cells touched by a 100 m direct flood-report buffer on a day",
               "not_target": "water depth or independently verified flooding",
               "train_dates": ["2022-01-01", "2025-07-31"],
               "calibration_dates": ["2025-08-01", "2025-09-30"],
               "test_dates": ["2025-10-01", "2025-12-31"],
               "input_features": ["month_sin", "month_cos", *RAIN_FEATURES],
               "calibration": "multiply daily count predictions by observed/predicted total in Aug–Sep",
               "districts": results,
               "limitation": "Exploratory reused test; district-point gridded rain, sparse complaint labels, no actual flood depths"}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUTPUT / f"{ARTIFACT}_daily.csv", daily_rows)


if __name__ == "__main__":
    main()
