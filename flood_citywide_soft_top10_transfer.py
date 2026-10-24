"""Train a shared transfer ranker with a differentiable top-10% queue loss.

The soft cutoff is the 90th percentile of sampled unreported cells from the
same day and source districts. Unreported means unknown, not observed dry.
Validation selects a checkpoint using only source-district dates before Q4.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from flood_bangkok_spatial_cnn_citywide import HOLDOUT
from flood_citywide_history_ranker import N_DAYS, TEST_START, TRAIN_START, VALID_START, load_grid
from flood_citywide_learned_history_transfer import (
    MODEL_NAMES, SEED, SharedHistoryRanker, evaluate, history_features, score_q4,
    static_features,
)
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_soft_top10_transfer.json"
OUT_MODEL = OUT / "q4_2025_soft_top10_transfer.pt"
USED_MODELS = (MODEL_NAMES[0], MODEL_NAMES[2])
REF_SIZE = 1024
POS_SIZE = 64


def examples(data: dict, static: np.ndarray, source: np.ndarray,
             days: range, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    references, positives, weights = [], [], []
    candidate_flat = data["candidate_flat"]
    for day in days:
        history, _ = history_features(data, day)
        features = np.concatenate((history, static), axis=1)
        truth = data["reports"][day].ravel()[candidate_flat].astype(bool)
        pos = np.flatnonzero(truth & source)
        if not len(pos):
            continue
        unknown = np.flatnonzero(~truth & source)
        ref = rng.choice(unknown, size=REF_SIZE, replace=False)
        chosen_positive = rng.choice(pos, size=POS_SIZE, replace=True)
        references.append(features[ref])
        positives.append(features[chosen_positive])
        weights.append(len(pos))
        if day % 90 == 0:
            print("prepared day", day, flush=True)
    return (torch.from_numpy(np.stack(references)),
            torch.from_numpy(np.stack(positives)),
            torch.tensor(weights, dtype=torch.float32))


def soft_queue_loss(model: SharedHistoryRanker, refs: torch.Tensor,
                    positives: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    batch = len(refs)
    ref_scores = model(refs.reshape(-1, refs.shape[-1])).reshape(batch, REF_SIZE)
    pos_scores = model(positives.reshape(-1, positives.shape[-1])).reshape(batch, POS_SIZE)
    cutoff = torch.quantile(ref_scores, .90, dim=1)
    scale = torch.clamp(ref_scores.std(dim=1), min=.3)
    margin = (cutoff[:, None]-pos_scores) / (.30*scale[:, None])
    day_loss = F.softplus(margin).mean(dim=1)
    return (day_loss*weights).sum()/weights.sum()


def fit(name: str, train: tuple, valid: tuple) -> tuple[SharedHistoryRanker, dict]:
    torch.manual_seed(SEED)
    model = SharedHistoryRanker(hidden=name != MODEL_NAMES[0],
                                physical=name == MODEL_NAMES[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    best, best_state, best_epoch = float("inf"), None, 0
    rng = np.random.default_rng(SEED)
    for epoch in range(1, 31):
        model.train()
        order = rng.permutation(len(train[0]))
        for start in range(0, len(order), 8):
            ids = order[start:start+8]
            loss = soft_queue_loss(model, train[0][ids], train[1][ids], train[2][ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            numerator, denominator = 0., 0.
            for start in range(0, len(valid[0]), 8):
                sl = slice(start, start+8)
                weight = float(valid[2][sl].sum())
                value = float(soft_queue_loss(model, valid[0][sl], valid[1][sl],
                                              valid[2][sl]))
                numerator += weight*value
                denominator += weight
            validation = numerator/denominator
        if validation < best-1e-4:
            best, best_epoch = validation, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0:
            print(name, "epoch", epoch, "source soft-top10 loss", round(validation, 5),
                  flush=True)
        if epoch-best_epoch >= 5:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "source_validation_soft_top10_loss": best}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_grid()
    district_ids = data["district_id"].ravel()[data["candidate_flat"]]
    heldout_ids = {i for i, district in enumerate(data["districts"])
                   if district["code"] in HOLDOUT}
    target = np.isin(district_ids, list(heldout_ids))
    source = ~target
    static, normalization = static_features(data, source)
    print("candidate/source/target", len(source), source.sum(), target.sum(), flush=True)
    train = examples(data, static, source, range(TRAIN_START, VALID_START), SEED)
    valid = examples(data, static, source, range(VALID_START, TEST_START), SEED+1)
    print("train days/positives", len(train[0]), train[2].sum().item(),
          "validation days/positives", len(valid[0]), valid[2].sum().item(), flush=True)
    models, fits = {}, {}
    for name in USED_MODELS:
        models[name], fits[name] = fit(name, train, valid)
    del train, valid
    scores, truth, no_prior = score_q4(data, static, models)
    masks = {"eight_unseen_districts": target, "other_42_districts": source,
             "all_bangkok": np.ones(len(source), bool)}
    q4 = evaluate(scores, truth, no_prior, masks)
    result = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "daily top-10% Traffy report-positive cell-day capture, not flooding",
        "training_period": ["2023-01-01", "2024-12-31"],
        "source_validation_period": ["2025-01-01", "2025-09-30"],
        "test_period": ["2025-10-01", "2025-12-31"],
        "heldout_district_codes": list(HOLDOUT),
        "loss": "softplus of sampled daily 90th-percentile unknown-cell score minus positive score; scores standardized by daily spread; positives weighted by daily count",
        "sampled_reference_cells_per_day": REF_SIZE,
        "sampled_positive_cells_per_day": POS_SIZE,
        "fits": fits, "q4": q4,
        "limitations": ["The soft cutoff approximates the full daily queue",
                        "Unreported comparison cells are unknown, not verified dry",
                        "Target districts retain their own prior-day report history",
                        "Q4 2025 was inspected in earlier project experiments",
                        "No rain or validated flood extent is used"],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2)+"\n")
    torch.save({"states": {name: model.state_dict() for name, model in models.items()},
                "normalization": normalization}, OUT_MODEL)
    for scope in masks:
        print(scope, {name: q4[scope][name]["top_10pct_capture"]
                      for name in ("raw_365d", "fixed_recent_nearby", *USED_MODELS)},
              flush=True)
    print("saved", OUT_JSON, OUT_MODEL, flush=True)


if __name__ == "__main__":
    main()
