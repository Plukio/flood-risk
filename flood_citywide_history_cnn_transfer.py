"""Spatial CNN residual on the daily Bangkok report-history queue.

The model reads 15x15 patches (250 m cells) of pre-day report counts, learns
one shared spatial correction to the fixed recent+nearby score, and is tested
in districts whose labels are excluded from fitting and validation.
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
from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, capture, load_grid,
)
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval

SEED = 5529
RADIUS = 7
REF_SIZE = 128
POS_SIZE = 32
MAX_EPOCHS = 8
OUT_JSON = OUT / "q4_2025_history_cnn_transfer.json"
OUT_MODEL = OUT / "q4_2025_history_cnn_transfer.pt"


def day_inputs(data: dict, day: int) -> tuple[np.ndarray, np.ndarray]:
    """No scored-day report enters an input channel or the fixed score."""
    prefix = data["prefix"]
    count = {k: (prefix[day]-prefix[day-k]).astype(np.float32)
             for k in (7, 30, 90, 365)}
    channels = [np.log1p(count[k]) for k in (365, 90, 30, 7)]
    channels.append(data["candidate"].astype(np.float32))
    base = (channels[0] + .4*channels[1] +
            .4*np.log1p(gaussian_filter(count[365], 1, mode="constant")))
    return np.stack(channels).astype(np.float32), base.astype(np.float32)


def patches(image: np.ndarray, candidate_flat: np.ndarray,
            indices: np.ndarray) -> torch.Tensor:
    rr, cc = np.unravel_index(candidate_flat[indices], SHAPE)
    offset = np.arange(-RADIUS, RADIUS+1)
    padded = np.pad(image, ((0, 0), (RADIUS, RADIUS),
                            (RADIUS, RADIUS)), mode="constant")
    picked = padded[:, rr[:, None, None]+RADIUS+offset[None, :, None],
                    cc[:, None, None]+RADIUS+offset[None, None, :]]
    return torch.from_numpy(np.ascontiguousarray(picked.transpose(1, 0, 2, 3)))


class SpatialResidualCNN(nn.Module):
    """Dilated 3x3 filters see out to seven 250 m cells in each direction."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 12, 3, padding=1), nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(12, 8, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def patch_scores(self, image_patches: torch.Tensor,
                     base: torch.Tensor) -> torch.Tensor:
        residual = self.net(image_patches)[:, 0, RADIUS, RADIUS]
        return base + residual

    def full_scores(self, image: torch.Tensor, base: np.ndarray,
                    candidate: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            padded = F.pad(image[None], (RADIUS, RADIUS, RADIUS, RADIUS))
            residual = self.net(padded)[0, 0, RADIUS:-RADIUS,
                                        RADIUS:-RADIUS].numpy()
        return (base+residual)[candidate]


def day_samples(data: dict, source: np.ndarray, days: range,
                seed: int) -> list[tuple[int, np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(seed)
    candidate_flat = data["candidate_flat"]
    result = []
    for day in days:
        truth = data["reports"][day].ravel()[candidate_flat].astype(bool)
        positive = np.flatnonzero(truth & source)
        if not len(positive):
            continue
        unknown = np.flatnonzero(~truth & source)
        ref = rng.choice(unknown, size=REF_SIZE, replace=False)
        pos = rng.choice(positive, size=POS_SIZE, replace=True)
        result.append((day, ref, pos, len(positive)))
    return result


def soft_queue_loss(model: SpatialResidualCNN, image: np.ndarray,
                    base: np.ndarray, candidate_flat: np.ndarray,
                    ref: np.ndarray, pos: np.ndarray) -> torch.Tensor:
    indices = np.concatenate((ref, pos))
    image_patches = patches(image, candidate_flat, indices)
    baseline = torch.from_numpy(base.ravel()[candidate_flat[indices]].copy())
    scores = model.patch_scores(image_patches, baseline)
    ref_scores, pos_scores = scores[:len(ref)], scores[len(ref):]
    cutoff = torch.quantile(ref_scores, .90)
    scale = torch.clamp(ref_scores.std(), min=.3)
    return F.softplus((cutoff-pos_scores)/(.30*scale)).mean()


def fit(data: dict, source: np.ndarray, seed: int) -> tuple[SpatialResidualCNN, dict]:
    train = day_samples(data, source, range(TRAIN_START, VALID_START), seed)
    valid = day_samples(data, source, range(VALID_START, TEST_START), seed+1)
    print("training/validation days", len(train), len(valid),
          "positives", sum(row[3] for row in train), sum(row[3] for row in valid),
          flush=True)
    torch.manual_seed(seed)
    model = SpatialResidualCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    mean_positive = np.mean([row[3] for row in train])
    best, best_state, best_epoch = float("inf"), None, 0
    rng = np.random.default_rng(seed)
    candidate_flat = data["candidate_flat"]
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        for index in rng.permutation(len(train)):
            day, ref, pos, count = train[index]
            image, base = day_inputs(data, day)
            loss = soft_queue_loss(model, image, base, candidate_flat, ref, pos)
            loss = loss*min(count/mean_positive, 3.)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        numerator, denominator = 0., 0
        with torch.no_grad():
            for day, ref, pos, count in valid:
                image, base = day_inputs(data, day)
                loss = soft_queue_loss(model, image, base, candidate_flat, ref, pos)
                numerator += count*float(loss)
                denominator += count
        value = numerator/denominator
        print("epoch", epoch, "source validation soft-top10 loss",
              round(value, 5), flush=True)
        if value < best-1e-4:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if epoch-best_epoch >= 3:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch,
                   "source_validation_soft_top10_loss": best,
                   "training_positive_cell_days": sum(row[3] for row in train),
                   "validation_positive_cell_days": sum(row[3] for row in valid)}


def score_q4(data: dict, model: SpatialResidualCNN) -> tuple[dict, np.ndarray]:
    candidate_flat = data["candidate_flat"]
    scores = {"raw_365d": [], "fixed_recent_nearby": [], "cnn_residual": []}
    truth = []
    for day in range(TEST_START, N_DAYS):
        image, base = day_inputs(data, day)
        local365 = np.expm1(image[0])[data["candidate"]]
        scores["raw_365d"].append(local365)
        scores["fixed_recent_nearby"].append(base[data["candidate"]])
        scores["cnn_residual"].append(model.full_scores(
            torch.from_numpy(image), base, data["candidate"]))
        truth.append(data["reports"][day].ravel()[candidate_flat].copy())
        if day % 30 == 0:
            print("scored Q4 day", day, flush=True)
    return {key: np.stack(value) for key, value in scores.items()}, np.stack(truth)


def evaluate(scores: dict, truth: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    result = {}
    for scope, mask in masks.items():
        rows = {name: capture(values, truth, mask) for name, values in scores.items()}
        rows["cnn_minus_fixed_95pct_weekly_block_interval"] = paired_interval(
            rows["cnn_residual"], rows["fixed_recent_nearby"])
        result[scope] = rows
    return result


def self_check() -> None:
    """The center patch must equal the same CNN at that point on a full map."""
    torch.manual_seed(SEED)
    model = SpatialResidualCNN()
    nn.init.normal_(model.net[-1].weight, std=.05)
    image = np.random.default_rng(SEED).normal(size=(5, *SHAPE)).astype(np.float32)
    base = np.zeros(SHAPE, np.float32)
    candidate_flat = np.arange(np.prod(SHAPE))
    chosen = np.array([0, 7*SHAPE[1]+7, 100*SHAPE[1]+100,
                       np.prod(SHAPE)-1])
    with torch.no_grad():
        patch_result = model.patch_scores(patches(image, candidate_flat, chosen),
                                          torch.zeros(len(chosen))).numpy()
        full_image = F.pad(torch.from_numpy(image[None]),
                           (RADIUS, RADIUS, RADIUS, RADIUS))
        full_result = model.net(full_image)[0, 0, RADIUS:-RADIUS,
                                                   RADIUS:-RADIUS].numpy().ravel()[chosen]
    if not np.allclose(patch_result, full_result, atol=1e-5):
        raise ValueError("Patch and full-grid CNN scores differ")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    self_check()
    data = load_grid()
    ids = data["district_id"].ravel()[data["candidate_flat"]]
    heldout_ids = {i for i, district in enumerate(data["districts"])
                   if district["code"] in HOLDOUT}
    target = np.isin(ids, list(heldout_ids))
    source = ~target
    model, fit_audit = fit(data, source, SEED)
    scores, truth = score_q4(data, model)
    masks = {"eight_unseen_districts": target, "other_42_districts": source,
             "all_bangkok": np.ones(len(source), bool)}
    result = evaluate(scores, truth, masks)
    if (result["all_bangkok"]["raw_365d"]["top_10pct_capture"] != .47183 or
        result["all_bangkok"]["fixed_recent_nearby"]["top_10pct_capture"] != .48168):
        raise ValueError("Baseline queue differs from frozen citywide evaluation")
    output = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "daily top-10% Traffy report-positive 250m cell-days, not flooding",
        "architecture": "15x15 pre-day report-history patch; 3 dilated 3x3 convolution layers; zero-initialized residual on fixed recent+nearby score; shared across districts",
        "history_channels_days": [365, 90, 30, 7],
        "candidate_mask_channel": True,
        "training_period": ["2023-01-01", "2024-12-31"],
        "source_validation_period": ["2025-01-01", "2025-09-30"],
        "test_period": ["2025-10-01", "2025-12-31"],
        "heldout_district_codes": list(HOLDOUT),
        "loss": "softplus of sampled daily 90th-percentile unknown-cell score minus positive score; normalized by daily score spread",
        "sampled_reference_cells_per_day": REF_SIZE,
        "sampled_positive_cells_per_day": POS_SIZE,
        "fit": fit_audit, "q4": result,
        "limitations": ["Unreported cells are unknown, not verified dry",
                        "Target districts retain own report history as features",
                        "Q4 2025 was inspected in previous project work",
                        "No rain or verified flood extent is used"],
    }
    OUT_JSON.write_text(json.dumps(output, indent=2)+"\n")
    torch.save({"state": model.state_dict(), "fit": fit_audit}, OUT_MODEL)
    for scope in masks:
        print(scope, {name: result[scope][name]["top_10pct_capture"]
                      for name in scores}, flush=True)
    print("saved", OUT_JSON, OUT_MODEL, flush=True)


if __name__ == "__main__":
    main()
