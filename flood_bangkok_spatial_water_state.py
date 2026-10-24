"""Test whether lagged river level changes cell ranking through local proxies.

The spatial connection here is only proximity to mapped canals and BMA
gate/pump inventory. It is not a verified hydraulic connection or gate state.
The same sampled-negative training design is used for all variants.

Run after flood_bangkok_shared_state_backtest.py:
    .venv/bin/python flood_bangkok_spatial_water_state.py
"""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss

from flood_bangkok_river_level_backtest import river_features, river_levels
from flood_bangkok_shared_state_backtest import preceding_wetness
from flood_bangkok_spatial_cnn_multiyear import CAL_END, TRAIN_END
from flood_bangkok_transfer import CACHE as TRANSFER_CACHE
from flood_bangkapi_surface_model import OUTPUT, rainfall_by_date, rank_metrics


CODES = ("1006", "1040")
ARTIFACT = "surface_spatial_water_state_backtest"
NEGATIVE_FRACTION = 0.20
BASE_FEATURES = [
    "elevation", "relative_elevation", "terrain_slope", "district_coverage",
    "nearest_pump_m", "pumps_within_1km", "nearest_gate_m",
    "mapped_canal_length_log", "nearest_canal_log", "prior_report_days_log",
    "month_sin", "month_cos", "rain_today_log", "rain_peak3h_log",
    "rain_prev72h_log", "rain_wet_hours_log", "rain_4_7d_log", "rain_8_14d_log",
]


def load_district(code: str, days: list[date], river: np.ndarray) -> dict:
    with np.load(OUTPUT / f"surface_shared_state_{code}_daily_cells.npz") as saved:
        y = saved["observed"]
        static = saved["static7"]
        rr, cc = saved["rows"], saved["cols"]
    if y.shape != (len(days), len(rr)) or static.shape != (len(rr), 7):
        raise ValueError(f"Unexpected cell cache shape for {code}")
    with np.load(TRANSFER_CACHE / f"canal_2025_{code}_250m.npz") as canal:
        if not np.all(canal["valid"][rr, cc]):
            raise ValueError(f"Canal grid does not cover {code}")
        canal_length = canal["canal_length_m"][rr, cc]
        canal_distance = canal["nearest_canal_m"][rr, cc]
    rain_path = TRANSFER_CACHE / f"rain_{code}_2021_2025.json"
    rain = rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31), rain_path)
    wet = preceding_wetness(days, rain_path)
    dynamic = np.asarray([[math.sin(2 * math.pi * day.month / 12),
                           math.cos(2 * math.pi * day.month / 12),
                           *(math.log1p(max(0, rain[day][key])) for key in (
                               "rain_today_mm", "rain_peak_3h_mm", "rain_prev72h_mm",
                               "rain_wet_hours")),
                           *wet[i]] for i, day in enumerate(days)], dtype=np.float32)
    prior = np.zeros(y.shape, dtype=np.uint16)
    for year in range(2022, 2026):
        indices = [i for i, day in enumerate(days) if day.year == year]
        annual = y[indices].astype(np.uint16)
        prior[indices] = np.cumsum(annual, axis=0) - annual
    n_cal = sum(day <= CAL_END for day in days)
    prior[n_cal:] = prior[n_cal]
    static_more = np.column_stack((static, np.log1p(canal_length),
                                   np.log1p(canal_distance))).astype(np.float32)
    gate_canal = (np.exp(-static[:, 6] / 1500) *
                  np.exp(-canal_distance / 500)).astype(np.float32)
    pump_canal = (np.exp(-static[:, 4] / 1500) *
                  np.exp(-canal_distance / 500)).astype(np.float32)
    return {"y": y, "static": static_more, "dynamic": dynamic, "prior": prior,
            "river": river[:, 0], "gate_canal": gate_canal,
            "pump_canal": pump_canal, "cells": len(rr)}


def matrix(info: dict, day_idx: np.ndarray, cell_idx: np.ndarray,
           variant: str) -> np.ndarray:
    x = np.column_stack((info["static"][cell_idx],
                         np.log1p(info["prior"][day_idx, cell_idx]),
                         info["dynamic"][day_idx])).astype(np.float32)
    if variant != "baseline":
        level = info["river"][day_idx]
        x = np.column_stack((x, level)).astype(np.float32)
        if variant == "river_with_drainage_interactions":
            x = np.column_stack((x, level * info["gate_canal"][cell_idx],
                                  level * info["pump_canal"][cell_idx])).astype(np.float32)
    return x


def training_indices(y: np.ndarray, train_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = y[:train_days].ravel()
    positive = np.flatnonzero(flat)
    negative = np.flatnonzero(flat == 0)
    rng = np.random.default_rng(7)
    selected = rng.choice(negative, size=round(len(negative) * NEGATIVE_FRACTION),
                          replace=False)
    flat_idx = np.concatenate((positive, selected))
    weights = np.concatenate((np.ones(len(positive), dtype=np.float32),
                              np.full(len(selected), 1 / NEGATIVE_FRACTION,
                                      dtype=np.float32)))
    return flat_idx // y.shape[1], flat_idx % y.shape[1], weights


def model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=100,
        max_leaf_nodes=12, min_samples_leaf=120, l2_regularization=20.0,
        max_bins=64, early_stopping=False, random_state=7)


def main() -> None:
    levels, _ = river_levels()
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   TRANSFER_CACHE / "rain_1006_2021_2025.json"))
    river = river_features(days, levels)
    data = {code: load_district(code, days, river) for code in CODES}
    n_train = sum(day <= TRAIN_END for day in days)
    n_cal = sum(day <= CAL_END for day in days)
    variants = ("baseline", "river_only", "river_with_drainage_interactions")
    comparisons = {}
    for name, source, target in (("bang_khae_local", "1040", "1040"),
                                 ("bang_kapi_to_bang_khae", "1006", "1040")):
        source_info, target_info = data[source], data[target]
        train_di, train_ci, weights = training_indices(source_info["y"], n_train)
        test_di = np.repeat(np.arange(n_cal, len(days)), target_info["cells"])
        test_ci = np.tile(np.arange(target_info["cells"]), len(days) - n_cal)
        truth = target_info["y"][n_cal:].astype(np.uint8)
        comparisons[name] = {}
        for variant in variants:
            x_train = matrix(source_info, train_di, train_ci, variant)
            x_test = matrix(target_info, test_di, test_ci, variant)
            target_train = source_info["y"][train_di, train_ci]
            fitted = model().fit(x_train, target_train, sample_weight=weights)
            probability = fitted.predict_proba(x_test)[:, 1].reshape(truth.shape)
            metrics = {
                "cell_rank": rank_metrics(probability.sum(axis=0), truth.sum(axis=0)),
                "daily_average_precision": round(float(average_precision_score(
                    truth.ravel(), probability.ravel())), 5),
                "daily_brier": round(float(brier_score_loss(
                    truth.ravel(), probability.ravel())), 6),
                "predicted_positive_cell_days": round(float(probability.sum()), 1),
                "observed_positive_cell_days": int(truth.sum()),
                "train_positives": int(target_train.sum()),
            }
            comparisons[name][variant] = metrics
            print(name, variant, metrics, flush=True)
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps({
        "target": "100 m flood-report buffer touches a 250 m cell-day; not verified inundation",
        "train_dates": ["2022-01-01", str(TRAIN_END)],
        "test_dates": ["2025-10-01", "2025-12-31"],
        "training": "All positive cell-days plus 20% of negatives with inverse-probability weights; identical sampled rows and model settings across variants",
        "feature_baseline": BASE_FEATURES,
        "river_only_adds": ["prior_day_Rama_VIII_river_max"],
        "river_interactions_add": ["river_level_x_gate_and_canal_proximity",
                                    "river_level_x_pump_and_canal_proximity"],
        "connection_limit": "Gate/pump and canal proximity is not proof of hydraulic connectivity; no historical operating state or paired water-level head is available",
        "missing_level": "Passed as NaN, including its interaction terms",
        "history_rule": "Previous report-positive days reset annually; test history freezes after September 2025",
        "rain_timing": "Full-day historical rain is used for explanatory hindcasting",
        "comparisons": comparisons,
        "limitation": "Single repeatedly inspected test season and two districts; source-to-target transfer remains exploratory"},
        ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
