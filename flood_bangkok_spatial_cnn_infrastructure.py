"""Backtest seven open-data infrastructure channels in the citywide CNN.

Run flood_bangkok_infrastructure_features.py first, then:
    .venv/bin/python flood_bangkok_spatial_cnn_infrastructure.py

Uses the identical 42-district fit, eight-district geographic holdout, and
2025 chronological blocks as the longer baseline citywide CNN. The label is
Traffy complaint occurrence, not verified flooding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flood_bangkok_infrastructure_features import FEATURE_DIR, FEATURES
from flood_bangkok_spatial_cnn_citywide import (
    ARTIFACT as BASE_ARTIFACT, CAL_N, CHANNELS, FEATURE_CACHE, HOLDOUT,
    cache_arrays, evaluate, prepare_cache, train_model,
)
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_model import OUTPUT, write_csv


ARTIFACT = "surface_crossdistrict_infrastructure_cnn_citywide_v1"
AUGMENTED_CACHE = CACHE / "cnn_citywide_infrastructure_daily_v1"
MODEL_NAME = "forty_two_infrastructure_long"


def augmented_arrays(meta: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    AUGMENTED_CACHE.mkdir(parents=True, exist_ok=True)
    base = cache_arrays(tuple(meta))
    result = {}
    for code in meta:
        raw_x, y = base[code]
        if raw_x.shape[1] != CHANNELS + 1:
            raise ValueError(f"Base feature count changed in {code}")
        with np.load(FEATURE_DIR / f"{code}.npz") as saved:
            values = saved["values"]
            rr, cc = saved["rows"], saved["cols"]
            names = tuple(saved["feature_names"].tolist())
        if names != FEATURES or not (np.array_equal(rr, meta[code]["valid_rows"]) and
                                      np.array_equal(cc, meta[code]["valid_cols"])):
            raise ValueError(f"Infrastructure features misaligned in {code}")
        h, w = raw_x.shape[2:]
        static = np.zeros((len(FEATURES), h, w), dtype=np.float16)
        static[:, rr, cc] = values.T.astype(np.float16)
        path = AUGMENTED_CACHE / f"{code}_x.npy"
        if not path.exists():
            partial = AUGMENTED_CACHE / f"{code}_x.partial.npy"
            augmented = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float16,
                shape=(len(raw_x), CHANNELS + len(FEATURES) + 1, h, w))
            for start in range(0, len(raw_x), 128):
                end = min(start + 128, len(raw_x))
                augmented[start:end, :CHANNELS] = raw_x[start:end, :CHANNELS]
                augmented[start:end, CHANNELS:CHANNELS + len(FEATURES)] = static
                augmented[start:end, -1] = raw_x[start:end, CHANNELS]
            augmented.flush()
            del augmented
            partial.replace(path)
        x = np.load(path, mmap_mode="r")
        if x.shape != (len(raw_x), CHANNELS + len(FEATURES) + 1, h, w):
            raise ValueError(f"Augmented cache shape mismatch in {code}")
        result[code] = x, y
        print(f"augmented {code} {x.shape}", flush=True)
    return result


def main() -> None:
    torch.set_num_threads(4)
    districts = {d["code"]: d for d in load_districts()}
    meta = prepare_cache(districts)
    arrays = augmented_arrays(meta)
    if int(np.asarray(arrays["1006"][1][CAL_N:]).sum()) != 242:
        raise ValueError("Bang Kapi labels changed")
    codes = tuple(code for code in districts if code not in HOLDOUT)
    baseline_path = OUTPUT / f"{BASE_ARTIFACT}_metrics.json"
    baseline = json.loads(baseline_path.read_text())
    reference = {row["code"]: row for row in baseline["models"][
        "forty_two_more_steps"]["evaluated_districts"]}
    model, stats, train = train_model(
        MODEL_NAME, codes, arrays, meta["1006"], max_epochs=40, patience=18,
        artifact=ARTIFACT)
    rows = evaluate(model, stats, arrays, meta, tuple(districts))
    eleven = set(baseline["models"]["eleven_matched"]["training"]["training_codes"])
    comparison = []
    for row in rows:
        code = row["code"]
        old = reference[code]
        if (row["test_positive_cell_days"] != old["test_positive_cell_days"] or
                row["cells"] != old["cells"] or
                row["historical_spearman"] != old["historical_spearman"]):
            raise ValueError(f"Baseline evaluation changed in {code}")
        comparison.append({
            "code": code, "district": row["district"],
            "group": "geographic_holdout" if code in HOLDOUT else
                     "original_eleven" if code in eleven else "added_training",
            "cells": row["cells"],
            "test_positive_cell_days": row["test_positive_cell_days"],
            "historical_spearman": row["historical_spearman"],
            "baseline_cnn_spearman": old["cnn_spearman"],
            "infrastructure_cnn_spearman": row["cnn_spearman"],
            "spearman_delta": round(row["cnn_spearman"] - old["cnn_spearman"], 4),
            "baseline_daily_ap": old["test_daily_ap"],
            "infrastructure_daily_ap": row["test_daily_ap"],
            "baseline_predicted_positive_cell_days": old["predicted_positive_cell_days"],
            "infrastructure_predicted_positive_cell_days": row["predicted_positive_cell_days"]})
    write_csv(OUTPUT / f"{ARTIFACT}_district_comparison.csv", comparison)
    payload = {
        "target": "100 m direct Traffy flood-complaint buffer touches 250 m cell-day",
        "not_target": "verified inundation or water depth",
        "features_added": list(FEATURES),
        "base_feature_count": CHANNELS,
        "infrastructure_source_audit": str(FEATURE_DIR / "audit.json"),
        "baseline_metrics": str(baseline_path),
        "train_period": baseline["train_period"],
        "selection_and_local_calibration": baseline["selection_and_local_calibration"],
        "test_period": baseline["test_period"],
        "geographic_holdout_codes": list(HOLDOUT),
        "training": train,
        "evaluated_districts": rows,
        "limitations": [
            "asset proximity does not establish a connected drain or outfall",
            "pump capacity units are not confirmed by the open catalog",
            "static 2021-2025 inventories are applied to earlier 2022-2024 observations",
            "underpass catalog has 11 usable geocoded sites out of 20 records",
            "report absence is not verified dry; 2025 test period is exploratory and reused"]}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("saved", ARTIFACT, "districts", len(rows), flush=True)


if __name__ == "__main__":
    main()
