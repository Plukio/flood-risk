"""Five-fold geographic transfer of the CNN residual with source-only calibration."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS, TEST_START, VALID_START, capture, load_grid
from flood_citywide_history_cnn_transfer import SEED, fit
from flood_citywide_history_cnn_alpha import ALPHAS, collect
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval

OUT_JSON = OUT / "q4_2025_history_cnn_fivefold_transfer.json"
OUT_MODEL = OUT / "q4_2025_history_cnn_fivefold_transfer.pt"


def aggregate(folds: list[dict], name: str) -> dict:
    pieces = [fold["q4"][name] for fold in folds]
    daily_positive = np.sum([row["daily_positive"] for row in pieces], axis=0)
    daily_captured = np.sum([row["daily_captured"] for row in pieces], axis=0)
    positives = int(daily_positive.sum())
    return {"candidate_cells": sum(row["candidate_cells"] for row in pieces),
            "positive_cell_days": positives,
            "top_10pct_capture": round(float(daily_captured.sum()/positives), 5),
            "daily_positive": daily_positive.astype(int).tolist(),
            "daily_captured": np.round(daily_captured, 5).tolist()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_grid()
    district_id = data["district_id"].ravel()[data["candidate_flat"]]
    district_folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    folds, states = [], {}
    for fold_number, ids in enumerate(district_folds, 1):
        target = np.isin(district_id, ids)
        source = ~target
        print("fold", fold_number, "target cells", target.sum(), flush=True)
        model, fit_audit = fit(data, source, SEED+fold_number*100)
        states[fold_number] = model.state_dict()
        val_base, val_residual, val_truth = collect(
            data, model, range(VALID_START, TEST_START))
        validation = {f"{alpha:g}": capture(val_base+alpha*val_residual,
                                             val_truth, source)["top_10pct_capture"]
                      for alpha in ALPHAS}
        chosen = max(ALPHAS, key=lambda alpha: validation[f"{alpha:g}"])
        del val_base, val_residual, val_truth
        base, residual, truth = collect(data, model, range(TEST_START, N_DAYS))
        q4 = {"fixed_recent_nearby": capture(base, truth, target),
              "cnn_selected": capture(base+chosen*residual, truth, target),
              "cnn_raw_alpha_1": capture(base+residual, truth, target)}
        fold = {"fold": fold_number,
                "district_codes": [data["districts"][i]["code"] for i in ids],
                "fit": fit_audit,
                "source_validation_capture_by_multiplier": validation,
                "selected_multiplier": chosen,
                "q4": q4,
                "selected_minus_fixed_95pct_weekly_block_interval":
                    paired_interval(q4["cnn_selected"], q4["fixed_recent_nearby"])}
        folds.append(fold)
        print("fold", fold_number, "alpha", chosen,
              {name: q4[name]["top_10pct_capture"] for name in q4}, flush=True)
    pooled = {name: aggregate(folds, name)
              for name in ("fixed_recent_nearby", "cnn_selected", "cnn_raw_alpha_1")}
    if (pooled["fixed_recent_nearby"]["candidate_cells"] != 25812 or
        pooled["fixed_recent_nearby"]["positive_cell_days"] != 7209):
        raise ValueError("Geographic folds do not cover frozen Bangkok Q4 queue")
    pooled["selected_minus_fixed_95pct_weekly_block_interval"] = paired_interval(
        pooled["cnn_selected"], pooled["fixed_recent_nearby"])
    output = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "daily top-10% Traffy report-positive 250m cell-days, not flooding",
        "architecture": "shared 15x15 dilated CNN residual on fixed recent+nearby score",
        "split": "same five seeded 10-district folds as learned-history rankers",
        "training_period": ["2023-01-01", "2024-12-31"],
        "source_validation_period": ["2025-01-01", "2025-09-30"],
        "test_period": ["2025-10-01", "2025-12-31"],
        "candidate_multipliers": ALPHAS,
        "queue_note": "Each fold has its own daily top-10% queue; pooled capture is not one citywide top-10% queue",
        "folds": folds, "pooled_fold_queues": pooled,
        "limitations": ["Unreported cells are unknown, not verified dry",
                        "Target districts retain own prior-day report history",
                        "Q4 2025 has been inspected in prior project work"],
    }
    OUT_JSON.write_text(json.dumps(output, indent=2)+"\n")
    torch.save({"states": states}, OUT_MODEL)
    print("pooled", {name: pooled[name]["top_10pct_capture"]
                     for name in ("fixed_recent_nearby", "cnn_selected",
                                  "cnn_raw_alpha_1")}, flush=True)
    print("saved", OUT_JSON, OUT_MODEL, flush=True)


if __name__ == "__main__":
    main()
