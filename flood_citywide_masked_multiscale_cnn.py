"""Center-masked multiscale report-history CNN; geographic transfer test.

The spatial branch cannot read the candidate cell's own report history. The
training cutoff is computed from every source-district candidate on that day.
Unreported cells are queue competitors, not verified dry observations.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import label
from torch import nn
from torch.nn import functional as F

from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, capture, load_grid,
)
from flood_citywide_history_cnn_alpha import ALPHAS
from flood_citywide_history_cnn_transfer import (
    SEED, SpatialResidualCNN, day_inputs, patches,
)
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval
from flood_open_weak_bangkok_holdout_eval import selected

OUT_JSON = OUT / "q4_2025_center_masked_multiscale_cnn_fivefold.json"
OUT_MODEL = OUT / "q4_2025_center_masked_multiscale_cnn_fivefold.pt"
OLD_MODEL = OUT / "q4_2025_history_cnn_fivefold_transfer.pt"
OLD_JSON = OUT / "q4_2025_history_cnn_fivefold_transfer.json"
MAX_EPOCHS = 4
POSITIVES_PER_DAY = 32


class MaskedMultiscaleCNN(nn.Module):
    """Three learned neighbor filters at 250 m, 750 m and 1.75 km reach."""

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(5, 8, 3, padding=d, dilation=d, bias=False)
            for d in (1, 3, 7)
        ])
        self.head = nn.Sequential(nn.Conv2d(24, 12, 1), nn.ReLU(),
                                  nn.Conv2d(12, 1, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        mask = torch.ones(1, 1, 3, 3)
        mask[:, :, 1, 1] = 0
        self.register_buffer("neighbor_mask", mask)

    def residual_map(self, image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F.relu(F.conv2d(image, branch.weight*self.neighbor_mask,
                                      None, padding=branch.padding,
                                      dilation=branch.dilation))
                       for branch in self.branches], dim=1)
        return self.head(x)[:, 0]

    def patch_scores(self, image_patches: torch.Tensor,
                     base: torch.Tensor) -> torch.Tensor:
        return base + self.residual_map(image_patches)[:, 7, 7]

    def full_scores(self, image: torch.Tensor, base: np.ndarray,
                    candidate: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            residual = self.residual_map(image[None])[0].numpy()
        return (base+residual)[candidate]


def self_check() -> None:
    torch.manual_seed(SEED)
    model = MaskedMultiscaleCNN()
    nn.init.normal_(model.head[-1].weight, std=.03)
    image = np.random.default_rng(SEED).normal(size=(5, *SHAPE)).astype(np.float32)
    flat = np.arange(np.prod(SHAPE))
    chosen = np.array([0, 7*SHAPE[1]+7, 100*SHAPE[1]+100,
                       np.prod(SHAPE)-1])
    with torch.no_grad():
        patch = model.patch_scores(patches(image, flat, chosen),
                                   torch.zeros(len(chosen))).numpy()
        full = model.residual_map(torch.from_numpy(image[None]))[0].numpy().ravel()[chosen]
        masked = image.copy()
        for index in chosen:
            rr, cc = np.unravel_index(index, SHAPE)
            masked[:, rr, cc] += 1000
            one = model.patch_scores(patches(masked, flat, np.array([index])),
                                     torch.zeros(1)).item()
            if not np.isclose(one, full[np.where(chosen == index)[0][0]], atol=1e-5):
                raise ValueError("Spatial branch reads center-cell input")
            masked[:, rr, cc] = image[:, rr, cc]
    if not np.allclose(patch, full, atol=1e-5):
        raise ValueError("Patch and full-map scores differ")


def cutoff_index(full_scores: np.ndarray, source_indices: np.ndarray) -> int:
    """One actual score at the top-10% queue boundary, with no subsampling."""
    subset = full_scores[source_indices]
    kth = len(subset)-math.ceil(.1*len(subset))
    return int(source_indices[np.argpartition(subset, kth)[kth]])


def fit(data: dict, source: np.ndarray, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MaskedMultiscaleCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.03)
    source_indices = np.flatnonzero(source)
    candidate_flat = data["candidate_flat"]
    train_days = [day for day in range(TRAIN_START, VALID_START)
                  if np.any(data["reports"][day].ravel()[candidate_flat[source_indices]])]
    best_value, best_state, best_epoch = -1., None, 0
    epochs = []
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        losses = []
        for day in rng.permutation(train_days):
            image, base = day_inputs(data, int(day))
            truth = data["reports"][day].ravel()[candidate_flat]
            positives = np.flatnonzero((truth > 0) & source)
            chosen = rng.choice(positives, size=min(POSITIVES_PER_DAY, len(positives)),
                                replace=False)
            # The queue boundary uses all source cells, then re-scores just
            # its boundary cell with gradient. No unknown is called dry.
            all_scores = model.full_scores(torch.from_numpy(image), base,
                                           data["candidate"])
            boundary = cutoff_index(all_scores, source_indices)
            indices = np.r_[chosen, boundary]
            local = patches(image, candidate_flat, indices)
            fixed = torch.from_numpy(base.ravel()[candidate_flat[indices]].copy())
            scores = model.patch_scores(local, fixed)
            margin = scores[-1]-scores[:-1]
            loss = F.softplus(margin/.4).mean()
            # Keep the spatial adjustment modest; a separate multiplier is
            # selected only on source-district validation queues.
            loss = loss + .002*(scores-fixed).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation = collect(data, model, range(VALID_START, TEST_START))
        source_capture = {f"{a:g}": capture(validation[0]+a*validation[1],
                                            validation[2], source)["top_10pct_capture"]
                          for a in ALPHAS}
        chosen_alpha = max(ALPHAS, key=lambda a: source_capture[f"{a:g}"])
        value = source_capture[f"{chosen_alpha:g}"]
        epochs.append({"epoch": epoch, "training_loss_mean": round(float(np.mean(losses)),5),
                       "source_validation_capture_by_multiplier": source_capture,
                       "selected_multiplier": chosen_alpha})
        print("epoch", epoch, "loss", round(float(np.mean(losses)),4),
              "source validation", value, "alpha", chosen_alpha, flush=True)
        if value > best_value+1e-5:
            best_value, best_epoch = value, epoch
            best_state = {k: v.detach().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch >= 2:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": epochs, "best_epoch": best_epoch,
                   "train_days_with_source_reports": len(train_days)}


def collect(data: dict, model: nn.Module, days: range):
    bases, residuals, truths, no_own = [], [], [], []
    candidate = data["candidate"]
    flat = data["candidate_flat"]
    for day in days:
        image, base = day_inputs(data, day)
        original = base[candidate].copy()
        prediction = model.full_scores(torch.from_numpy(image), base, candidate)
        bases.append(original)
        residuals.append(prediction-original)
        truths.append(data["reports"][day].ravel()[flat].copy())
        no_own.append(data["prefix"][day].ravel()[flat] ==
                      data["prefix"][day-365].ravel()[flat])
    return np.stack(bases), np.stack(residuals), np.stack(truths), np.stack(no_own)


def diagnostic(scores: np.ndarray, truth: np.ndarray, no_own: np.ndarray,
               mask: np.ndarray, candidate: np.ndarray) -> dict:
    """Same full queue for all strata; adjacent same-day cells proxy clusters."""
    indices = np.flatnonzero(mask)
    no_own_positive = no_own[:, indices] & (truth[:, indices] > 0)
    own_positive = int(no_own_positive.sum())
    no_own_hits = 0.
    clusters = 0
    cluster_hits = 0.
    daily_clusters, daily_cluster_hits = [], []
    full_flat = np.flatnonzero(candidate.ravel())[indices]
    for i in range(len(scores)):
        picks = selected(scores[i, indices], .1)
        no_own_hits += float(np.dot(picks, no_own_positive[i]))
        grid = np.zeros(SHAPE, bool)
        grid.ravel()[full_flat] = truth[i, indices] > 0
        components, n = label(grid, structure=np.ones((3,3), bool))
        ids = components.ravel()[full_flat]
        hit = 0.
        for cluster_id in range(1, n+1):
            # Ties are fractional inspection capacity; max is the expected
            # chance at least one member is selected under tie allocation.
            hit += float(np.max(picks[ids == cluster_id]))
        clusters += n
        cluster_hits += hit
        daily_clusters.append(n)
        daily_cluster_hits.append(round(hit, 5))
    return {"no_own_365d_positive_cell_days": own_positive,
            "no_own_365d_capture_at_full_queue": round(no_own_hits/own_positive,5)
                if own_positive else None,
            "same_day_adjacent_cell_clusters": clusters,
            "cluster_capture_at_full_queue": round(cluster_hits/clusters,5)
                if clusters else None,
            "daily_clusters": daily_clusters,
            "daily_cluster_hits": daily_cluster_hits}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    self_check()
    data = load_grid()
    district = data["district_id"].ravel()[data["candidate_flat"]]
    district_folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    old_artifact = json.loads(OLD_JSON.read_text())
    old_states = torch.load(OLD_MODEL, weights_only=True)["states"]
    folds, states = [], {}
    for fold_number, ids in enumerate(district_folds, 1):
        target = np.isin(district, ids)
        source = ~target
        print("fold", fold_number, "target cells", target.sum(), flush=True)
        model, fit_audit = fit(data, source, SEED+fold_number*100)
        states[fold_number] = {k:v.cpu() for k,v in model.state_dict().items()}
        val_base, val_residual, val_truth, _ = collect(
            data, model, range(VALID_START, TEST_START))
        validation = {f"{a:g}": capture(val_base+a*val_residual,
                                         val_truth, source)["top_10pct_capture"]
                      for a in ALPHAS}
        alpha = max(ALPHAS, key=lambda a: validation[f"{a:g}"])
        del val_base, val_residual, val_truth
        base, residual, truth, no_own = collect(data, model,
                                                 range(TEST_START, N_DAYS))
        old = SpatialResidualCNN()
        old.load_state_dict(old_states[fold_number])
        old.eval()
        _, old_residual, _, _ = collect(data, old, range(TEST_START, N_DAYS))
        old_alpha = old_artifact["folds"][fold_number-1]["selected_multiplier"]
        all_scores = {"fixed_recent_nearby": base,
                      "old_cnn_selected": base+old_alpha*old_residual,
                      "masked_multiscale_cnn_selected": base+alpha*residual}
        q4 = {name: capture(values, truth, target)
              for name,values in all_scores.items()}
        diag = {name: diagnostic(values, truth, no_own, target,
                                 data["candidate"])
                for name,values in all_scores.items()}
        folds.append({"fold": fold_number,
                      "district_codes": [data["districts"][j]["code"] for j in ids],
                      "fit": fit_audit,
                      "source_validation_capture_by_multiplier": validation,
                      "selected_multiplier": alpha,
                      "old_cnn_selected_multiplier": old_alpha,
                      "q4": q4, "diagnostic": diag,
                      "masked_minus_fixed_weekly_block_interval": paired_interval(
                          q4["masked_multiscale_cnn_selected"],
                          q4["fixed_recent_nearby"])})
        print("fold", fold_number, "alpha", alpha,
              {k:v["top_10pct_capture"] for k,v in q4.items()}, flush=True)
    names = ("fixed_recent_nearby", "old_cnn_selected",
             "masked_multiscale_cnn_selected")
    pooled = {}
    for name in names:
        rows = [f["q4"][name] for f in folds]
        positives = np.sum([r["daily_positive"] for r in rows], axis=0)
        hits = np.sum([r["daily_captured"] for r in rows], axis=0)
        details = [f["diagnostic"][name] for f in folds]
        unseen_positive = sum(r["no_own_365d_positive_cell_days"] for r in details)
        unseen_hits = sum(r["no_own_365d_capture_at_full_queue"]*
                          r["no_own_365d_positive_cell_days"] for r in details)
        clusters = sum(r["same_day_adjacent_cell_clusters"] for r in details)
        cluster_hits = sum(sum(r["daily_cluster_hits"]) for r in details)
        pooled[name] = {"candidate_cells": sum(r["candidate_cells"] for r in rows),
                        "positive_cell_days": int(positives.sum()),
                        "top_10pct_capture": round(float(hits.sum()/positives.sum()),5),
                        "daily_positive": positives.astype(int).tolist(),
                        "daily_captured": np.round(hits,5).tolist(),
                        "no_own_365d_positive_cell_days": unseen_positive,
                        "no_own_365d_capture_at_full_queue": round(unseen_hits/unseen_positive,5),
                        "same_day_adjacent_cell_clusters": clusters,
                        "cluster_capture_at_full_queue": round(cluster_hits/clusters,5)}
    if (pooled["fixed_recent_nearby"]["candidate_cells"] != 25812 or
        pooled["fixed_recent_nearby"]["positive_cell_days"] != 7209 or
        pooled["fixed_recent_nearby"]["top_10pct_capture"] != .47113):
        raise ValueError("Matched five-fold baseline changed")
    old_expected = old_artifact["pooled_fold_queues"]["cnn_selected"]["top_10pct_capture"]
    if abs(pooled["old_cnn_selected"]["top_10pct_capture"]-old_expected) > 1e-5:
        raise ValueError("Old CNN did not reproduce on matched queues")
    pooled["masked_minus_fixed_weekly_block_interval"] = paired_interval(
        pooled["masked_multiscale_cnn_selected"], pooled["fixed_recent_nearby"])
    pooled["masked_minus_old_cnn_weekly_block_interval"] = paired_interval(
        pooled["masked_multiscale_cnn_selected"], pooled["old_cnn_selected"])
    output = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "target": "daily top-10% Traffy report-positive 250m cell-days, not flooding",
              "architecture": "center-masked 3-branch CNN at dilation 1, 3, 7; residual on fixed recent+nearby history",
              "training_cutoff": "actual top-10% boundary over every source-district candidate each day; boundary cell and sampled positives receive gradient",
              "split": "same five seeded 10-district folds as earlier CNN",
              "training_period": ["2023-01-01", "2024-12-31"],
              "source_validation_period": ["2025-01-01", "2025-09-30"],
              "test_period": ["2025-10-01", "2025-12-31"],
              "folds": folds, "pooled_fold_queues": pooled,
              "cluster_definition": "8-connected report-positive 250m cells on the same date within each held-out fold; proxy, not original complaint IDs",
              "limitations": ["Unreported cells are unknown, not verified dry",
                              "Q4 2025 inspected in prior project work",
                              "Target districts retain prior-day reports as input features",
                              "No rain, hydraulic flow or verified extent in this model"]}
    OUT_JSON.write_text(json.dumps(output, indent=2)+"\n")
    torch.save({"states": states}, OUT_MODEL)
    print("pooled", {k:pooled[k]["top_10pct_capture"] for k in names}, flush=True)
    print("saved", OUT_JSON, OUT_MODEL, flush=True)


if __name__ == "__main__":
    main()
