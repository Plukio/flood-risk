"""Adapt Bang Khae CNN scores to local report history and daily counts.

The source CNN supplies within-day spatial scores. A Bang Khae count model,
fitted before August 2025 and level-calibrated on August–September, supplies
the expected number of report-positive cells per day. A common daily log-odds
shift makes cell probabilities sum to that count, preserving each day's cell
ordering. A fixed 50/50 logit blend with January–September local report
history checks whether a local spatial prior helps geographic transfer.
October–December 2025 is a reused exploratory test. These remain
Traffy report-proxy probabilities, not verified inundation probabilities.

Run after flood_bangkok_spatial_cnn_transfer_test.py and
flood_bangkok_local_rate_calibration.py:
    .venv/bin/python flood_bangkok_spatial_rate_hybrid.py
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from flood_bangkapi_proxy_probability import evaluation
from flood_bangkapi_surface_model import OUTPUT, rank_metrics, write_csv


TRANSFER = "surface_crossdistrict_terrain_cnn_bangkhae_transfer"
COUNT = "surface_local_rain_count_calibration_2025"
ARTIFACT = "surface_bangkhae_spatial_rate_hybrid"


def match_daily_logits(logits: np.ndarray, count: float) -> np.ndarray:
    if not (0 < count < len(logits)):
        raise ValueError("District count must be inside the number of cells")
    bias = brentq(lambda shift: float(expit(logits + shift).sum() - count),
                  -40, 40)
    return expit(logits + bias)


def match_daily_count(probability: np.ndarray, count: float) -> np.ndarray:
    return match_daily_logits(logit(np.clip(probability, 1e-10, 1 - 1e-10)), count)


def main() -> None:
    with (OUTPUT / f"{COUNT}_daily.csv").open(encoding="utf-8-sig", newline="") as file:
        daily = {}
        for row in csv.DictReader(file):
            if row["district_code"] == "1040":
                if row["date"] in daily:
                    raise ValueError(f"Duplicate local count date: {row['date']}")
                daily[row["date"]] = row
    dates = [date(2025, 10, 1) + timedelta(days=i) for i in range(92)]
    if len(daily) != 92 or {str(day) for day in dates} != set(daily):
        raise ValueError("Local count predictions do not match the 92 test days")
    count_model = np.asarray([float(daily[str(day)][
        "calibrated_predicted_positive_cell_days"]) for day in dates])
    recent_rate = np.asarray([float(daily[str(day)][
        "recent_rate_predicted_positive_cell_days"]) for day in dates])
    result = {"test_district": "Bang Khae", "test_dates": [str(dates[0]), str(dates[-1])],
              "count_model_reference": f"{COUNT}_metrics.json",
              "transfer_model_reference": f"{TRANSFER}_metrics.json", "models": {}}
    with (OUTPUT / f"{TRANSFER}_cell_comparison.csv").open(
            encoding="utf-8-sig", newline="") as file:
        original_rows = list(csv.DictReader(file))
    rows = [{"cell_id": row["cell_id"],
             "historical_positive_days_jan_sep": int(row["historical_positive_days_jan_sep"]),
             "observed_days_oct_dec": int(row["observed_days_oct_dec"])}
            for row in original_rows]
    if len(rows) != 840 or len({row["cell_id"] for row in rows}) != 840:
        raise ValueError("Expected 840 unique Bang Khae cells")
    historical = np.asarray([row["historical_positive_days_jan_sep"] for row in rows])
    prior_logits = logit((historical + 0.5) / 274.0)
    prior_logits -= prior_logits.mean()
    for name in ("ten_districts_2025_only", "ten_districts_multiyear"):
        with np.load(OUTPUT / f"{TRANSFER}_{name}_daily.npz") as saved:
            source = saved["source_calibrated"]
            adapted = saved["target_recalibrated"]
            truth = saved["observed"]
        if source.shape != (92, 840) or truth.shape != source.shape:
            raise ValueError("Unexpected daily transfer prediction shape")
        if not np.array_equal(truth.sum(axis=0),
                              np.asarray([r["observed_days_oct_dec"] for r in rows])):
            raise ValueError("Daily truth disagrees with comparison table")
        if not np.isfinite(source).all() or not np.isfinite(adapted).all():
            raise ValueError("CNN predictions contain nonfinite values")
        source_logits = logit(np.clip(source, 1e-10, 1 - 1e-10))
        centered_source = source_logits - source_logits.mean(axis=1, keepdims=True)
        candidates = {"source_calibrated": source,
                      "target_intercept": adapted,
                      "local_rain_count": np.stack([
                          match_daily_count(source[i], count_model[i]) for i in range(92)]),
                      "recent_rate_count": np.stack([
                          match_daily_count(source[i], recent_rate[i]) for i in range(92)]),
                      "local_rain_count_prior_blend": np.stack([
                          match_daily_logits(0.5 * centered_source[i] +
                                             0.5 * prior_logits, count_model[i])
                          for i in range(92)]),
                      "local_rain_count_prior_only": np.stack([
                          match_daily_logits(prior_logits, count_model[i])
                          for i in range(92)])}
        result["models"][name] = {}
        for variant, p in candidates.items():
            expected = p.sum(axis=0)
            result["models"][name][variant] = {
                "test_daily": evaluation(p.ravel(), truth.ravel()),
                "test_cell_rank": rank_metrics(expected, truth.sum(axis=0)),
                "daily_count_mae": round(float(np.mean(np.abs(
                    p.sum(axis=1) - truth.sum(axis=1)))), 4)}
            for i, row in enumerate(rows):
                row[f"{name}_{variant}_expected_days"] = round(float(expected[i]), 5)
    result["local_count_model_test_total"] = round(float(count_model.sum()), 1)
    result["recent_rate_test_total"] = round(float(recent_rate.sum()), 1)
    result["observed_test_total"] = int(sum(row["observed_days_oct_dec"] for row in rows))
    result["method"] = ("per-day common log-odds shift matches local count estimate; "
                        "source-only within-day spatial rank unchanged; "
                        "local prior uses Jan–Sep 2025 report counts with 0.5 smoothing "
                        "and fixed 50/50 centered-logit blend")
    result["limitation"] = "exploratory reused test; local count and cell labels are complaint proxies, not independent water observations"
    write_csv(OUTPUT / f"{ARTIFACT}_cell_comparison.csv", rows)
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        result, ensure_ascii=False, indent=2) + "\n")
    for name, variants in result["models"].items():
        for variant, metrics in variants.items():
            print(name, variant, metrics["test_cell_rank"]["spearman"],
                  metrics["test_daily"]["average_precision"],
                  metrics["test_daily"]["predicted_positive_cell_days"])


if __name__ == "__main__":
    main()
