"""Apply the existing Bang Khae rain-count correction to canal CNN scores.

The daily log-odds shift preserves within-day canal-CNN cell ordering. It uses
only pre-October local count fitting/calibration and actual test-period rain.
October–December 2025 is a reused exploratory report-proxy evaluation.

Run after flood_bangkok_spatial_cnn_canals.py and the local count model:
    .venv/bin/python flood_bangkok_canal_rate_hybrid.py
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta

import numpy as np

from flood_bangkok_spatial_rate_hybrid import match_daily_count
from flood_bangkapi_proxy_probability import evaluation
from flood_bangkapi_surface_model import OUTPUT, rank_metrics, write_csv


CANAL = "surface_crossdistrict_bangkhae_canal_cnn"
ARTIFACT = "surface_crossdistrict_bangkhae_canal_rate_hybrid"


def main() -> None:
    dates = [date(2025, 10, 1) + timedelta(days=i) for i in range(92)]
    with (OUTPUT / "surface_local_rain_count_calibration_2025_daily.csv").open(
            encoding="utf-8-sig", newline="") as file:
        by_date = {}
        for row in csv.DictReader(file):
            if row["district_code"] == "1040":
                if row["date"] in by_date:
                    raise ValueError("Duplicate Bang Khae count date")
                by_date[row["date"]] = row
    if {str(day) for day in dates} != set(by_date):
        raise ValueError("Bang Khae count dates do not match the test")
    counts = np.asarray([float(by_date[str(day)][
        "calibrated_predicted_positive_cell_days"]) for day in dates])
    with np.load(OUTPUT / f"{CANAL}_daily.npz") as saved:
        score = saved["target_recalibrated"]
        observed = saved["observed"]
    if score.shape != observed.shape or score.shape != (92, 840):
        raise ValueError("Unexpected canal CNN prediction shape")
    corrected = np.stack([match_daily_count(score[i], counts[i]) for i in range(92)])
    original = json.loads((OUTPUT / "surface_bangkhae_spatial_rate_hybrid_metrics.json").read_text())[
        "models"]["ten_districts_multiyear"]["local_rain_count"]
    variants = {}
    for name, probability in (("canal_cnn_target_intercept", score),
                              ("canal_cnn_local_rain_count", corrected)):
        variants[name] = {
            "test_daily": evaluation(probability.ravel(), observed.ravel()),
            "test_cell_rank": rank_metrics(probability.sum(axis=0), observed.sum(axis=0)),
            "daily_count_mae": round(float(np.mean(np.abs(
                probability.sum(axis=1) - observed.sum(axis=1)))), 4)}
    with (OUTPUT / f"{CANAL}_cell_comparison.csv").open(
            encoding="utf-8-sig", newline="") as file:
        source_rows = list(csv.DictReader(file))
    cell_ids = [row["cell_id"] for row in source_rows]
    if not np.array_equal(observed.sum(axis=0), np.asarray([
            int(row["observed_days_oct_dec"]) for row in source_rows])):
        raise ValueError("Canal CNN cell table and daily labels disagree")
    rows = [{"cell_id": cell_id,
             "mapped_canal_length_in_cell_m": source_rows[i][
                 "mapped_canal_length_in_cell_m"],
             "distance_to_mapped_canal_m": source_rows[i][
                 "distance_to_mapped_canal_m"],
             "canal_cnn_expected_days": round(float(score[:, i].sum()), 6),
             "canal_hybrid_expected_days": round(float(corrected[:, i].sum()), 6),
             "observed_days_oct_dec": int(observed[:, i].sum())}
            for i, cell_id in enumerate(cell_ids)]
    if len(rows) != 840 or len({row["cell_id"] for row in rows}) != 840:
        raise ValueError("Canal cell table must have 840 unique cells")
    reference = {"test_daily": original["test_daily"],
                 "test_cell_rank": original["test_cell_rank"],
                 "daily_count_mae": original["daily_count_mae"]}
    result = {"target": "100 m Traffy flood-report buffer, not verified inundation",
              "test_dates": [str(dates[0]), str(dates[-1])],
              "canal_source": f"{CANAL}_metrics.json",
              "count_source": "surface_local_rain_count_calibration_2025_metrics.json",
              "baseline_no_canal_cnn_local_rain_count": reference,
              "variants": variants,
              "limitation": "Exploratory reused test; count correction does not validate physical canal hydraulics"}
    write_csv(OUTPUT / f"{ARTIFACT}_cell_comparison.csv", rows)
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        result, ensure_ascii=False, indent=2) + "\n")
    for name, item in variants.items():
        print(name, item["test_cell_rank"]["spearman"],
              item["test_daily"]["average_precision"],
              item["test_daily"]["predicted_positive_cell_days"])


if __name__ == "__main__":
    main()
