"""Transfer test for a learned Bangkok report-history ranker.

Target: Traffy report-positive 250 m cell-days, not verified flooding.
All history precedes the scored day. Unreported comparisons are unknown.
Run: .venv/bin/python flood_citywide_learned_history_transfer.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F

from flood_bangkok_spatial_cnn_citywide import HOLDOUT
from flood_citywide_history_ranker import N_DAYS, TEST_START, TRAIN_START, VALID_START, capture, load_grid
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval
from flood_open_weak_data import FEATURE_NAMES

SEED = 8412
HISTORY_NAMES = (
    "local_365d", "local_90d", "local_30d", "local_7d",
    "smooth_365d_250m", "smooth_90d_250m", "smooth_30d_250m",
    "smooth_365d_750m", "smooth_90d_750m", "smooth_365d_1500m",
)
STATIC_NAMES = (
    "srtm_relative_1250m_m", "srtm_slope",
    "osm_canal_length_log", "osm_canal_distance_log",
    "osm_road_length_log", "osm_road_distance_log",
    "worldcover_built_fraction_2021", "worldcover_crop_fraction_2021",
    "worldcover_tree_grass_fraction_2021",
)
MODEL_NAMES = ("learned_linear_history", "monotone_history_mlp",
               "monotone_history_plus_static")
OUT_JSON = OUT / "q4_2025_learned_history_transfer.json"
OUT_MODEL = OUT / "q4_2025_learned_history_transfer.pt"


def history_features(data: dict, day: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = data["prefix"]
    counts = {k: (prefix[day] - prefix[day-k]).astype(np.float32)
              for k in (7, 30, 90, 365)}
    mask = data["candidate"]
    maps = [counts[k] for k in (365, 90, 30, 7)]
    maps += [gaussian_filter(counts[k], 1, mode="constant") for k in (365, 90, 30)]
    maps += [gaussian_filter(counts[k], 3, mode="constant") for k in (365, 90)]
    maps += [gaussian_filter(counts[365], 6, mode="constant")]
    x = np.stack([np.log1p(m[mask]) for m in maps], axis=1)
    return x.astype(np.float32), counts[365][mask]


def fixed_score(history: np.ndarray) -> np.ndarray:
    return history[:, 0] + .4*history[:, 1] + .4*history[:, 4]


def static_features(data: dict, source_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    indices = [FEATURE_NAMES.index(name) for name in STATIC_NAMES]
    raw = data["static"][:, indices].astype(np.float32)
    mean = raw[source_mask].mean(axis=0)
    std = np.maximum(raw[source_mask].std(axis=0), .05)
    norm = np.clip((raw-mean)/std, -5, 5).astype(np.float32)
    return norm, {"mean": mean.tolist(), "std": std.tolist()}


def sample_pairs(data: dict, static: np.ndarray, source_mask: np.ndarray,
                 days: range, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    pos_rows, cmp_rows = [], []
    total_positive = 0
    candidate_flat = data["candidate_flat"]
    for day in days:
        history, _ = history_features(data, day)
        truth = data["reports"][day].ravel()[candidate_flat].astype(bool)
        positive = np.flatnonzero(truth & source_mask)
        if not len(positive):
            continue
        unknown = np.flatnonzero(~truth & source_mask)
        base = fixed_score(history)
        order = np.argsort(base[unknown])
        top = unknown[order[-max(100, int(.10*len(unknown))):]]
        boundary = unknown[order[int(.75*len(unknown)):int(.90*len(unknown))]]
        # Same-day pairwise ranking; unreported cells are not verified dry.
        cmp = np.concatenate((
            rng.choice(unknown, size=(len(positive), 4)),
            rng.choice(top, size=(len(positive), 2)),
            rng.choice(boundary, size=(len(positive), 2))), axis=1).ravel()
        pos = np.repeat(positive, 8)
        features = np.concatenate((history, static), axis=1)
        pos_rows.append(features[pos])
        cmp_rows.append(features[cmp])
        total_positive += len(positive)
        if day % 90 == 0:
            print("sampled through day", day, flush=True)
    return (np.concatenate(pos_rows), np.concatenate(cmp_rows),
            {"positive_cell_days": total_positive, "comparison_pairs": 8*total_positive})


def inverse_softplus(values: np.ndarray) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(values, dtype=torch.float32)))


class SharedHistoryRanker(nn.Module):
    """Shared monotone history response plus optional static residual."""

    def __init__(self, hidden: bool, physical: bool):
        super().__init__()
        start = np.full(len(HISTORY_NAMES), .01, np.float32)
        start[[0, 1, 4]] = [1., .4, .4]
        self.raw_direct = nn.Parameter(inverse_softplus(start))
        self.hidden = hidden
        self.physical = physical
        if hidden:
            self.raw_in = nn.Parameter(-3. + .1*torch.randn(len(HISTORY_NAMES), 12))
            self.hidden_bias = nn.Parameter(-2. + .1*torch.randn(12))
            self.raw_out = nn.Parameter(-5. + .1*torch.randn(12))
        if physical:
            self.static_net = nn.Sequential(nn.Linear(len(STATIC_NAMES), 16),
                                            nn.Tanh(), nn.Linear(16, 1))
            nn.init.normal_(self.static_net[0].weight, std=.03)
            nn.init.zeros_(self.static_net[0].bias)
            nn.init.zeros_(self.static_net[2].weight)
            nn.init.zeros_(self.static_net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history = x[:, :len(HISTORY_NAMES)]
        score = history @ F.softplus(self.raw_direct)
        if self.hidden:
            hidden = F.softplus(history @ F.softplus(self.raw_in) + self.hidden_bias)
            score = score + hidden @ F.softplus(self.raw_out)
        if self.physical:
            score = score + self.static_net(x[:, len(HISTORY_NAMES):])[:, 0]
        return score


def pair_loss(model: SharedHistoryRanker, positive: torch.Tensor,
              comparison: torch.Tensor) -> torch.Tensor:
    return F.softplus(model(comparison)-model(positive)).mean()


def fit_model(name: str, train: tuple[np.ndarray, np.ndarray],
              valid: tuple[np.ndarray, np.ndarray]) -> tuple[SharedHistoryRanker, dict]:
    torch.manual_seed(SEED)
    model = SharedHistoryRanker(hidden=name != MODEL_NAMES[0],
                                physical=name == MODEL_NAMES[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.006, weight_decay=.005)
    positive, comparison = map(torch.from_numpy, train)
    vpos, vcmp = map(torch.from_numpy, valid)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    rng = np.random.default_rng(SEED)
    for epoch in range(1, 71):
        model.train()
        order = rng.permutation(len(positive))
        for start in range(0, len(order), 8192):
            ids = order[start:start+8192]
            loss = pair_loss(model, positive[ids], comparison[ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = [pair_loss(model, vpos[i:i+8192], vcmp[i:i+8192]).item()
                      for i in range(0, len(vpos), 8192)]
        validation = float(np.mean(losses))
        if validation < best_loss-1e-5:
            best_loss, best_epoch = validation, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0:
            print(name, "epoch", epoch, "validation pair loss", round(validation, 5), flush=True)
        if epoch-best_epoch >= 7:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    direct = F.softplus(model.raw_direct).detach().numpy().tolist()
    return model, {"best_epoch": best_epoch, "source_validation_pair_loss": best_loss,
                   "direct_history_weights": dict(zip(HISTORY_NAMES, direct))}


def score_q4(data: dict, static: np.ndarray,
             models: dict[str, SharedHistoryRanker]) -> tuple[dict, np.ndarray, np.ndarray]:
    scores = {"raw_365d": [], "fixed_recent_nearby": []}
    scores.update({name: [] for name in models})
    truth_rows, no_prior_rows = [], []
    candidate_flat = data["candidate_flat"]
    for day in range(TEST_START, N_DAYS):
        history, prior_count = history_features(data, day)
        features = torch.from_numpy(np.concatenate((history, static), axis=1))
        scores["raw_365d"].append(prior_count.copy())
        scores["fixed_recent_nearby"].append(fixed_score(history))
        with torch.no_grad():
            for name, model in models.items():
                scores[name].append(model(features).numpy())
        truth = data["reports"][day].ravel()[candidate_flat].copy()
        truth_rows.append(truth)
        no_prior_rows.append(truth * (prior_count == 0))
        if day % 30 == 0:
            print("scored Q4 day", day, flush=True)
    return ({name: np.stack(rows) for name, rows in scores.items()},
            np.stack(truth_rows), np.stack(no_prior_rows))


def evaluate(scores: dict, truth: np.ndarray, no_prior: np.ndarray,
             masks: dict[str, np.ndarray]) -> dict:
    results = {}
    for scope, mask in masks.items():
        rows = {name: capture(values, truth, mask) for name, values in scores.items()}
        # Keep the same top-10% daily queue but count only positives at cells
        # without their own reported incident during the preceding year.
        unseen = {name: capture(values, no_prior, mask) for name, values in scores.items()}
        rows["no_prior_report_capture"] = {
            name: {k: row[k] for k in ("candidate_cells", "positive_cell_days",
                                      "top_10pct_capture")}
            for name, row in unseen.items()}
        rows["versus_fixed_95pct_weekly_block_interval"] = {
            name: paired_interval(rows[name], rows["fixed_recent_nearby"])
            for name in scores if name in MODEL_NAMES}
        results[scope] = rows
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_grid()
    district_ids = data["district_id"].ravel()[data["candidate_flat"]]
    holdout_ids = {i for i, district in enumerate(data["districts"])
                   if district["code"] in HOLDOUT}
    target = np.isin(district_ids, list(holdout_ids))
    source = ~target
    static, normalization = static_features(data, source)
    print("candidate/source/target", len(source), source.sum(), target.sum(), flush=True)
    train_pos, train_cmp, train_audit = sample_pairs(
        data, static, source, range(TRAIN_START, VALID_START), SEED)
    valid_pos, valid_cmp, valid_audit = sample_pairs(
        data, static, source, range(VALID_START, TEST_START), SEED+1)
    print("training", train_audit, "validation", valid_audit, flush=True)
    models, fits = {}, {}
    for name in MODEL_NAMES:
        models[name], fits[name] = fit_model(name, (train_pos, train_cmp),
                                            (valid_pos, valid_cmp))
    del train_pos, train_cmp, valid_pos, valid_cmp
    scores, truth, no_prior = score_q4(data, static, models)
    masks = {"eight_unseen_districts": target, "other_42_districts": source,
             "all_bangkok": np.ones(len(source), bool),
             "outside_old_patches_750m": data["new"].ravel()[data["candidate_flat"]]}
    result = evaluate(scores, truth, no_prior, masks)
    if (result["all_bangkok"]["raw_365d"]["top_10pct_capture"] != .47183 or
        result["all_bangkok"]["fixed_recent_nearby"]["top_10pct_capture"] != .48168):
        raise ValueError("Baseline labels or queue differ from frozen citywide backtest")
    output = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "daily top-10% Traffy report-positive 250m cell-day capture, not flooding",
        "training_period": ["2023-01-01", "2024-12-31"],
        "source_validation_period": ["2025-01-01", "2025-09-30"],
        "test_period": ["2025-10-01", "2025-12-31"],
        "heldout_district_codes": list(HOLDOUT),
        "history_features": HISTORY_NAMES, "static_features": STATIC_NAMES,
        "static_normalization_from_source": normalization,
        "loss": "same-day pairwise logistic: positives vs sampled unreported cells; half uniform, one quarter top-10% hard, one quarter near top-10% boundary",
        "architecture": "shared monotone positive-weight history ranker; optional 12-unit monotone hidden layer and 16-unit static residual; no district IDs or rain",
        "train_pairs": train_audit, "source_validation_pairs": valid_audit,
        "fits": fits, "q4": result,
        "limitations": ["Unreported cells are unknown, not verified dry",
                        "Held-out districts retain their own pre-day report history",
                        "Q4 2025 was inspected in earlier project experiments",
                        "Static 2021 cover and OSM/SRTM snapshots may not match event dates",
                        "Ranking scores are not flood probabilities or rain scenarios"],
    }
    OUT_JSON.write_text(json.dumps(output, indent=2) + "\n")
    torch.save({"states": {name: model.state_dict() for name, model in models.items()},
                "normalization": normalization, "history_features": HISTORY_NAMES,
                "static_features": STATIC_NAMES}, OUT_MODEL)
    for scope in masks:
        print(scope, {name: result[scope][name]["top_10pct_capture"]
                      for name in ("raw_365d", "fixed_recent_nearby", *MODEL_NAMES)}, flush=True)
    print("saved", OUT_JSON, OUT_MODEL, flush=True)


if __name__ == "__main__":
    main()
