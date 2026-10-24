"""Five-fold district transfer check for the learned-history ranking experiment.

Each model is fit on 40 districts in 2023-24, chosen with 2025 Jan-Sep
source-district pairs, and scored on the other 10 districts in Q4 2025.
Within each fold, every method gets the same daily 10% candidate budget.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS, TEST_START, TRAIN_START, VALID_START, capture, load_grid
from flood_citywide_learned_history_transfer import (
    MODEL_NAMES, SEED, fit_model, sample_pairs, score_q4, static_features,
)
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval

OUT_JSON = OUT / "q4_2025_learned_history_fivefold_transfer.json"
USED_MODELS = (MODEL_NAMES[0], MODEL_NAMES[2])
SCORE_NAMES = ("raw_365d", "fixed_recent_nearby", *USED_MODELS)


def aggregate(folds: list[dict], label: str) -> dict:
    rows = {}
    for name in SCORE_NAMES:
        pieces = [fold[label][name] for fold in folds]
        positive = sum(row["positive_cell_days"] for row in pieces)
        captured = np.sum([row["daily_captured"] for row in pieces], axis=0)
        daily_positive = np.sum([row["daily_positive"] for row in pieces], axis=0)
        rows[name] = {
            "candidate_cells": sum(row["candidate_cells"] for row in pieces),
            "positive_cell_days": positive,
            "top_10pct_capture": round(float(captured.sum()/positive), 5),
            "daily_positive": daily_positive.astype(int).tolist(),
            "daily_captured": np.round(captured, 5).tolist(),
        }
    rows["versus_fixed_95pct_weekly_block_interval"] = {
        name: paired_interval(rows[name], rows["fixed_recent_nearby"])
        for name in USED_MODELS}
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_grid()
    districts = data["districts"]
    district_ids = data["district_id"].ravel()[data["candidate_flat"]]
    rng = np.random.default_rng(917)
    fold_ids = np.array_split(rng.permutation(len(districts)), 5)
    folds = []
    for fold_number, ids in enumerate(fold_ids, 1):
        target = np.isin(district_ids, ids)
        source = ~target
        static, normalization = static_features(data, source)
        print("fold", fold_number, "target cells", target.sum(), flush=True)
        train_pos, train_cmp, train_audit = sample_pairs(
            data, static, source, range(TRAIN_START, VALID_START), SEED+fold_number*10)
        valid_pos, valid_cmp, valid_audit = sample_pairs(
            data, static, source, range(VALID_START, TEST_START), SEED+fold_number*10+1)
        models, fits = {}, {}
        for name in USED_MODELS:
            models[name], fits[name] = fit_model(name, (train_pos, train_cmp),
                                                (valid_pos, valid_cmp))
        del train_pos, train_cmp, valid_pos, valid_cmp
        scores, truth, no_prior = score_q4(data, static, models)
        q4 = {name: capture(values, truth, target) for name, values in scores.items()}
        new_sites = {name: capture(values, no_prior, target) for name, values in scores.items()}
        fold = {
            "fold": fold_number,
            "district_codes": [districts[i]["code"] for i in ids],
            "train_pairs": train_audit, "source_validation_pairs": valid_audit,
            "static_normalization_from_source": normalization,
            "fits": fits, "q4": q4, "no_prior_reports": new_sites,
            "versus_fixed_95pct_weekly_block_interval": {
                name: paired_interval(q4[name], q4["fixed_recent_nearby"])
                for name in USED_MODELS},
        }
        folds.append(fold)
        print("fold", fold_number, {name: q4[name]["top_10pct_capture"]
                                  for name in SCORE_NAMES}, flush=True)
    combined = aggregate(folds, "q4")
    combined_no_prior = aggregate(folds, "no_prior_reports")
    if (combined["raw_365d"]["positive_cell_days"] != 7209 or
        combined["raw_365d"]["candidate_cells"] != 25812):
        raise ValueError("Fold partition failed to cover frozen Q4 candidate queue")
    result = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "daily top-10% Traffy report-positive cell-day capture, not verified flooding",
        "split": "five nonoverlapping seeded 10-district folds; source 40 districts; target 10",
        "training_period": ["2023-01-01", "2024-12-31"],
        "source_validation_period": ["2025-01-01", "2025-09-30"],
        "test_period": ["2025-10-01", "2025-12-31"],
        "queue_note": "Each fold has its own daily top-10% queue; weighted aggregate is not one citywide top-10% queue",
        "folds": folds, "pooled_fold_queues": combined,
        "pooled_fold_queues_no_prior_reports": combined_no_prior,
        "limitations": ["Unreported cells are unknown, not verified dry",
                        "Each target district retains its own prior-day complaint history",
                        "Q4 2025 has been inspected in previous project work",
                        "Static layers can differ from event dates"],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print("pooled", {name: combined[name]["top_10pct_capture"]
                     for name in SCORE_NAMES}, flush=True)
    print("saved", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
