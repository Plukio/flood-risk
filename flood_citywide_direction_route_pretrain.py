"""Direction-aware version of synthetic D8 route pretraining.

Each cell receives eight supervised incoming-flow components, one for each
neighbor direction, plus its outgoing D8 neighbor or sink class. Total-arrival,
directional-flux, directional-share, and outgoing-direction losses train the
spatial trunk before the same isolated-site fine-tuning as
flood_citywide_route_pretrain.py.

Run: .venv/bin/python flood_citywide_direction_route_pretrain.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, VALID_START, load_grid
from flood_citywide_route_pretrain import (
    DECAY, HOPS, OUT, finetune, lidar_static, pooled, route_self_check,
    scenario, score_days, terrain,
)
from flood_citywide_spatial_novel_queue import SpatialNovelCNN


OUT_JSON = OUT / "q4_2025_direction_route_outgoing_pretrain_isolated_fivefold.json"
OUT_MODEL = OUT / "q4_2025_direction_route_outgoing_pretrain_isolated_fivefold.pt"
FLUX_WEIGHT = .5
DIRECTION_WEIGHT = .1
OUTGOING_WEIGHT = 1.


def incoming_direction(node_flat: np.ndarray, down: np.ndarray) -> np.ndarray:
    """For every D8 edge, encode where its source lies relative to receiver."""
    result = np.full(len(down), -1, np.int8)
    valid = down >= 0
    src_r, src_c = np.unravel_index(node_flat[valid], SHAPE)
    dst_r, dst_c = np.unravel_index(node_flat[down[valid]], SHAPE)
    dr, dc = src_r-dst_r, src_c-dst_c
    if np.any(np.maximum(abs(dr), abs(dc)) != 1):
        raise ValueError("D8 edge is not between immediate neighbors")
    code = (dr+1)*3 + (dc+1)
    if np.any(code == 4):
        raise ValueError("D8 self edge")
    result[valid] = code - (code > 4)
    return result


def outgoing_direction(node_flat: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Nine-class downhill-neighbor label, including an explicit sink class."""
    result = np.full(SHAPE, -1, np.int8)
    labels = np.full(len(node_flat), 8, np.int8)
    valid = down >= 0
    src_r, src_c = np.unravel_index(node_flat[valid], SHAPE)
    dst_r, dst_c = np.unravel_index(node_flat[down[valid]], SHAPE)
    dr, dc = dst_r-src_r, dst_c-src_c
    code = (dr+1)*3 + (dc+1)
    if np.any(np.maximum(abs(dr), abs(dc)) != 1) or np.any(code == 4):
        raise ValueError("Invalid outgoing D8 edge")
    labels[valid] = code-(code > 4)
    result.ravel()[node_flat] = labels
    return result


def directional_route(source: np.ndarray, node_flat: np.ndarray,
                      down: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Flux at each receiver, resolved by the eight incoming directions."""
    current = source.ravel()[node_flat].astype(np.float32).copy()
    incoming = np.zeros((8, len(node_flat)), np.float32)
    valid = down >= 0
    for _ in range(HOPS):
        next_flux = DECAY*current[valid]
        following = np.zeros_like(current)
        np.add.at(following, down[valid], next_flux)
        np.add.at(incoming, (direction[valid], down[valid]), next_flux)
        current = following
    result = np.zeros((8, *SHAPE), np.float32)
    result.reshape(8, -1)[:, node_flat] = incoming
    return result


def direction_self_check() -> None:
    source = np.zeros(SHAPE, np.float32)
    source[0, 0] = 1.
    nodes = np.asarray([0, 1, 2], np.int32)
    down = np.asarray([1, 2, -1], np.int32)
    ids = incoming_direction(nodes, down)
    got = directional_route(source, nodes, down, ids)
    if (not np.isclose(got[:, 0, 1].sum(), DECAY) or
        not np.isclose(got[:, 0, 2].sum(), DECAY**2) or
        not np.array_equal(np.flatnonzero(got[:, 0, 1]), [3]) or
        np.count_nonzero(got) != 2):
        raise ValueError("Incoming-direction routing changed")
    outgoing = outgoing_direction(nodes, down)
    if not np.array_equal(outgoing.ravel()[:3], [4, 4, 8]):
        raise ValueError("Outgoing-direction classes changed")


class DirectionRouteCNN(SpatialNovelCNN):
    """Shared CNN trunk with an eight-direction nonnegative flux head."""

    def __init__(self):
        super().__init__()
        self.flux_head = nn.Conv2d(8, 8, 1)
        nn.init.normal_(self.flux_head.weight, std=.02)
        nn.init.constant_(self.flux_head.bias, -3.)
        self.outgoing_head = nn.Conv2d(8, 9, 1)

    def patch_pretext(self, batch: torch.Tensor):
        valid = batch[:, -1:]
        features = F.relu(self.entry(batch))*valid
        for step in self.steps:
            update = step["mix"](F.relu(step["spatial"](features)))
            features = F.relu(features + .25*update)*valid
        return (F.softplus(self.flux_head(features)[:, :, RADIUS, RADIUS]),
                self.outgoing_head(features)[:, :, RADIUS, RADIUS])


def direction_loss(pred_flux: torch.Tensor, true_flux: torch.Tensor,
                   outgoing_logits: torch.Tensor,
                   outgoing_target: torch.Tensor):
    """Tie total arrival to its per-edge decomposition and main direction."""
    predicted_total = torch.log1p(pred_flux.sum(dim=1))
    true_total = torch.log1p(true_flux.sum(dim=1))
    total = F.smooth_l1_loss(predicted_total, true_total)
    flux = F.smooth_l1_loss(pred_flux, true_flux, reduction="none").sum(1).mean()
    active = true_flux.sum(dim=1) > .02
    if not torch.any(active):
        raise ValueError("No active routed target in batch")
    share = true_flux[active] / true_flux[active].sum(1, keepdim=True)
    direction = -(share*F.log_softmax(
        torch.log(pred_flux[active]+1e-6), dim=1)).sum(1).mean()
    has_terrain = outgoing_target >= 0
    outgoing = F.cross_entropy(outgoing_logits[has_terrain],
                               outgoing_target[has_terrain].long())
    objective = (total + FLUX_WEIGHT*flux + DIRECTION_WEIGHT*direction +
                 OUTGOING_WEIGHT*outgoing)
    return objective, (total, flux, direction, outgoing)


def pretrain_direction(data: dict, source: np.ndarray, static: np.ndarray,
                       node_flat: np.ndarray, down: np.ndarray,
                       direction: np.ndarray, outgoing: np.ndarray, seed: int,
                       steps: int, batch_size: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 491)
    model = DirectionRouteCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    scope = np.zeros(SHAPE, bool)
    scope.ravel()[data["candidate_flat"]] = source
    built = np.zeros(SHAPE, np.float32)
    built[data["candidate"]] = np.clip(data["static"][:, 7], 0, 1)
    source_flat = data["candidate_flat"][source]
    losses = []
    model.train()
    for step in range(steps):
        image, target = scenario(rng, data, static, scope, built,
                                 node_flat, down)
        flux = directional_route(image[0], node_flat, down, direction)
        if not np.allclose(np.log1p(flux.sum(0)), target, atol=1e-5):
            raise ValueError("Directional and scalar route targets disagree")
        values = target.ravel()[source_flat]
        active = np.flatnonzero(values > .02)
        inactive = np.flatnonzero(values <= .02)
        if not len(active) or not len(inactive):
            raise ValueError("Synthetic scenario has no active/inactive sites")
        chosen = np.r_[rng.choice(active, batch_size//2, replace=True),
                       rng.choice(inactive, batch_size-batch_size//2,
                                  replace=True)]
        batch = patches(image, source_flat, chosen)
        true_flux = torch.from_numpy(np.ascontiguousarray(
            flux.reshape(8, -1)[:, source_flat[chosen]].T))
        predicted, outgoing_logits = model.patch_pretext(batch)
        outgoing_target = torch.from_numpy(outgoing.ravel()[source_flat[chosen]].copy())
        loss, components = direction_loss(predicted, true_flux,
                                          outgoing_logits, outgoing_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append([float(loss.detach()),
                       *[float(c.detach()) for c in components]])
        if (step+1) % 40 == 0:
            print("direction step", step+1, "losses",
                  np.round(np.mean(losses[-40:], axis=0), 5).tolist(),
                  flush=True)
    model.eval()
    scope[:] = False
    scope.ravel()[data["candidate_flat"]] = ~source
    heldout_flat = data["candidate_flat"][~source]
    prediction, truth, pred_dir, true_dir = [], [], [], []
    pred_out, true_out = [], []
    with torch.no_grad():
        for _ in range(3):
            image, target = scenario(rng, data, static, scope, built,
                                     node_flat, down)
            flux = directional_route(image[0], node_flat, down, direction)
            chosen = rng.choice(len(heldout_flat),
                                min(2000, len(heldout_flat)), replace=False)
            pred_tensor, out_logits = model.patch_pretext(
                patches(image, heldout_flat, chosen))
            pred = pred_tensor.numpy()
            actual = flux.reshape(8, -1)[:, heldout_flat[chosen]].T
            prediction.append(np.log1p(pred.sum(1)))
            truth.append(target.ravel()[heldout_flat[chosen]])
            active = actual.sum(1) > .02
            pred_dir.extend(pred[active].argmax(1).tolist())
            true_dir.extend(actual[active].argmax(1).tolist())
            out_truth = outgoing.ravel()[heldout_flat[chosen]]
            out_valid = out_truth >= 0
            pred_out.extend(out_logits.numpy()[out_valid].argmax(1).tolist())
            true_out.extend(out_truth[out_valid].tolist())
    prediction, truth = np.concatenate(prediction), np.concatenate(truth)
    true_dir, pred_dir = np.asarray(true_dir), np.asarray(pred_dir)
    majority = np.bincount(true_dir, minlength=8).max()/len(true_dir)
    true_out, pred_out = np.asarray(true_out), np.asarray(pred_out)
    outgoing_majority = np.bincount(true_out, minlength=9).max()/len(true_out)
    loss_array = np.asarray(losses)
    fit = {"steps": steps,
           "loss_components": ["total_objective", "arrival_huber",
                               "sum_direction_flux_huber", "direction_share_ce",
                               "outgoing_d8_ce"],
           "loss_first_20": np.round(loss_array[:20].mean(0), 5).tolist(),
           "loss_last_20": np.round(loss_array[-20:].mean(0), 5).tolist(),
           "heldout_synthetic_spearman": round(float(spearmanr(
               prediction, truth).statistic), 5),
           "heldout_synthetic_mae": round(float(np.mean(np.abs(
               prediction-truth))), 5),
           "heldout_active_direction_top1": round(float(np.mean(
               pred_dir == true_dir)), 5),
           "heldout_active_majority_direction_top1": round(float(majority), 5),
           "heldout_active_count": len(true_dir),
           "heldout_outgoing_top1": round(float(np.mean(pred_out == true_out)), 5),
           "heldout_outgoing_majority_top1": round(float(outgoing_majority), 5),
           "heldout_outgoing_count": len(true_out)}
    return model, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-steps", type=int, default=200)
    parser.add_argument("--pretrain-batch", type=int, default=192)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    if args.pretrain_steps < 20 or args.pretrain_batch < 16:
        raise ValueError("Too few synthetic training examples")
    route_self_check()
    direction_self_check()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_grid()
    elevation, node_flat, down, terrain_audit = terrain(data)
    direction = incoming_direction(node_flat, down)
    outgoing = outgoing_direction(node_flat, down)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    records, states = [], {}
    for fold_id, ids in enumerate(folds, 1):
        source = ~np.isin(district, ids)
        static, norm = lidar_static(data, source, elevation)
        seed = SEED + fold_id*1000 + args.seed_offset
        print("fold", fold_id, "source", int(source.sum()), flush=True)
        pre, synthetic_fit = pretrain_direction(
            data, source, static, node_flat, down, direction, outgoing,
            seed, args.pretrain_steps, args.pretrain_batch)
        tuned, report_fit = finetune(data, source, static, seed, pre)
        models = {"direction_route_pretrained_cnn": tuned}
        validation = score_days(data, source, static, models,
                                range(VALID_START, TEST_START))
        test = score_days(data, ~source, static, models,
                          range(TEST_START, N_DAYS))
        records.append({"fold": fold_id,
                        "heldout_district_codes": [data["districts"][j]["code"]
                                                   for j in ids],
                        "normalization": norm,
                        "synthetic_fit": synthetic_fit,
                        "report_fit": report_fit,
                        "source_validation": validation,
                        "heldout_q4": test})
        states[fold_id] = {k: v.cpu() for k, v in tuned.state_dict().items()}
        print("fold", fold_id, {name: (r["isolated_hits"],
                                     r["group_hits"]) for name, r in
                                     test.items()}, flush=True)
    result = {"built_utc": datetime.now(timezone.utc).isoformat(
                  timespec="seconds"),
              "question": "Does direction-aware route pretraining transfer better than scalar-arrival pretraining?",
              "synthetic_target": "Eight incoming D8 neighbor fluxes per cell, summed over 1-5 downhill hops",
              "synthetic_loss": "Huber(log1p total arrival) + 0.5 * sum-of-eight flux Huber + 0.1 * incoming flux-share cross-entropy on active cells + 1.0 * nine-class outgoing D8 direction cross-entropy",
              "flux_weight": FLUX_WEIGHT,
              "direction_weight": DIRECTION_WEIGHT,
              "outgoing_weight": OUTGOING_WEIGHT,
              "training_dates": ["2023-01-01", "2024-12-31"],
              "source_validation_dates": ["2025-01-01", "2025-09-30"],
              "geographic_test_dates": ["2025-10-01", "2025-12-31"],
              "queue": "10% per heldout fold per day; 20% reserve for strictly isolated candidates",
              "terrain": terrain_audit,
              "pretrain_steps": args.pretrain_steps,
              "pretrain_batch": args.pretrain_batch,
              "seed_offset": args.seed_offset,
              "folds": records,
              "pooled_source_validation": pooled(records, "source_validation"),
              "pooled_heldout_q4": pooled(records, "heldout_q4"),
              "limitations": ["Synthetic D8 labels are terrain rules, not observed water flow",
                              "Unreported report cells are unknown, not verified dry",
                              "Canal hydraulic direction and drainage infrastructure are absent from synthetic labels",
                              "Present-day cached LiDAR is not a 2023-2025 vintage-matched archive",
                              "Q4 2025 has been inspected repeatedly; results are exploratory"]}
    if (result["pooled_heldout_q4"]["distant_history"]["isolated_positive"] != 332 or
        result["pooled_heldout_q4"]["distant_history"]["isolated_hits"] != 65.):
        raise ValueError("Frozen isolated queue changed")
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"states": states}, OUT_MODEL)
    print("pooled", {k: (v["isolated_hits"], v["group_hits"],
                         v["all_capture"]) for k, v in
                         result["pooled_heldout_q4"].items()}, flush=True)
    print("saved", OUT_JSON, OUT_MODEL, flush=True)


if __name__ == "__main__":
    main()
