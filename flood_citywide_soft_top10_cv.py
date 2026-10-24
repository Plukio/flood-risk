"""Five-fold district transfer check for the soft top-10% learned ranker."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS, TEST_START, TRAIN_START, VALID_START, capture, load_grid
from flood_citywide_learned_history_transfer import SEED, score_q4, static_features
from flood_citywide_learned_history_cv import aggregate, SCORE_NAMES
from flood_citywide_soft_top10_transfer import USED_MODELS, examples, fit
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval

OUT_JSON = OUT / "q4_2025_soft_top10_fivefold_transfer.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_grid()
    districts = data["districts"]
    ids_by_cell = data["district_id"].ravel()[data["candidate_flat"]]
    folds = []
    for fold_number, ids in enumerate(np.array_split(
            np.random.default_rng(917).permutation(len(districts)), 5), 1):
        target = np.isin(ids_by_cell, ids)
        source = ~target
        static, normalization = static_features(data, source)
        print("fold", fold_number, "target cells", target.sum(), flush=True)
        train = examples(data, static, source, range(TRAIN_START, VALID_START),
                         SEED+fold_number*10)
        valid = examples(data, static, source, range(VALID_START, TEST_START),
                         SEED+fold_number*10+1)
        models, fits = {}, {}
        for name in USED_MODELS:
            models[name], fits[name] = fit(name, train, valid)
        del train, valid
        scores, truth, no_prior = score_q4(data, static, models)
        q4 = {name: capture(values, truth, target) for name, values in scores.items()}
        no_prior_q4 = {name: capture(values, no_prior, target)
                       for name, values in scores.items()}
        folds.append({
            "fold": fold_number,
            "district_codes": [districts[i]["code"] for i in ids],
            "static_normalization_from_source": normalization,
            "fits": fits, "q4": q4, "no_prior_reports": no_prior_q4,
            "versus_fixed_95pct_weekly_block_interval": {
                name: paired_interval(q4[name], q4["fixed_recent_nearby"])
                for name in USED_MODELS},
        })
        print("fold", fold_number, {name: q4[name]["top_10pct_capture"]
                                  for name in SCORE_NAMES}, flush=True)
    pooled = aggregate(folds, "q4")
    pooled_no_prior = aggregate(folds, "no_prior_reports")
    if (pooled["raw_365d"]["candidate_cells"] != 25812 or
        pooled["raw_365d"]["positive_cell_days"] != 7209):
        raise ValueError("District folds do not cover the frozen Bangkok test queue")
    output = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "daily top-10% Traffy report-positive cell-day capture, not flooding",
        "split": "same five seeded district folds as pairwise experiment",
        "training_period": ["2023-01-01", "2024-12-31"],
        "source_validation_period": ["2025-01-01", "2025-09-30"],
        "test_period": ["2025-10-01", "2025-12-31"],
        "loss": "softplus of sampled daily 90th-percentile unknown-cell score minus positive score; normalized by daily score spread",
        "queue_note": "Separate top-10% queue within each fold; pooled result is not a single citywide queue",
        "folds": folds, "pooled_fold_queues": pooled,
        "pooled_fold_queues_no_prior_reports": pooled_no_prior,
        "limitations": ["Unreported is unknown, not verified dry",
                        "Target districts retain their own prior-day report history",
                        "Q4 2025 was inspected in prior project work",
                        "No rain or verified flood extent was used"],
    }
    OUT_JSON.write_text(json.dumps(output, indent=2)+"\n")
    print("pooled", {name: pooled[name]["top_10pct_capture"]
                     for name in SCORE_NAMES}, flush=True)
    print("saved", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
