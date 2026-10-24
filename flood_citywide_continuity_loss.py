"""Fine-tune the saved masked report CNN with a local continuity proxy.

The score is a report-site rank, not flood depth or calibrated probability.
This auxiliary loss only discourages a low-scored center when at least three
side neighbors score higher and the coarse DTM makes flow into the center
plausible. It does not turn unreported cells into observed dry labels.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid
from flood_citywide_route_pretrain import terrain
from flood_citywide_spatial_mask_augmentation import OUT_MODEL, mask_random_blocks
from flood_citywide_spatial_novel_queue import MASK_RATE, SpatialNovelCNN, collect, day_maps, static_image
from flood_citywide_target_site_mask_test import q20_rows
from flood_multi_target_citywide_data import OUT

OUTPUT = OUT / "q4_2025_mask_cnn_continuity_loss_fivefold.json"
STATES = OUT / "q4_2025_mask_cnn_continuity_loss_fivefold.pt"
ARMS = ("no_aux", "ordinary", "terrain")
LAMBDA = 1.0
MARGIN = .1  # ranking-logit units, not flood probability
MAX_GAP_M = .25
LR = .00025
EPOCHS = 1


def gates(data: dict, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Side-neighbor potential openings; missing barrier records prove nothing."""
    flat = data["candidate_flat"]
    rr, cc = np.unravel_index(flat, SHAPE)
    ordinary = np.zeros((len(flat), 4), bool)
    physical = np.zeros_like(ordinary)
    neighbor_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    center_z = z.ravel()[flat]
    for direction, (dr, dc) in enumerate(neighbor_offsets):
        valid = (rr + dr >= 0) & (rr + dr < SHAPE[0]) & (cc + dc >= 0) & (cc + dc < SHAPE[1])
        source = np.flatnonzero(valid)
        dst = flat[source] + dr * SHAPE[1] + dc
        connected = data["candidate"].ravel()[dst]
        source, dst = source[connected], dst[connected]
        next_z = z.ravel()[dst]
        finite = np.isfinite(center_z[source]) & np.isfinite(next_z)
        ordinary[source[finite], direction] = True
        # A low center is a plausible receiving cell when three neighbors
        # are no more than 25 cm above it. Mean height cannot certify passage.
        plausible = finite & (next_z >= center_z[source]) & \
                    (next_z-center_z[source] <= MAX_GAP_M)
        physical[source[plausible], direction] = True
    audit = {
        "candidate_cells": len(flat),
        "ordinary_centers_with_3_neighbors": int((ordinary.sum(axis=1) >= 3).sum()),
        "terrain_centers_with_3_potential_incoming_neighbors": int((physical.sum(axis=1) >= 3).sum()),
        "terrain_rule": "center mean DTM <= neighbor mean DTM <= center + 0.25 m; side-adjacent candidate cells",
        "barriers": "unknown, not confirmed absent",
    }
    return ordinary, physical, audit


def continuity_loss(local_logits: torch.Tensor, gate: torch.Tensor,
                    margin: float = MARGIN) -> tuple[torch.Tensor, int, float]:
    """Penalize a low center score below the third-highest eligible neighbor."""
    c = RADIUS
    neighbors = torch.stack((local_logits[:, c-1, c], local_logits[:, c+1, c],
                             local_logits[:, c, c-1], local_logits[:, c, c+1]), dim=1)
    center_score = local_logits[:, c, c]
    eligible = gate.sum(dim=1) >= 3
    if not bool(eligible.any()):
        return center_score.sum()*0, 0, 0.0
    third = neighbors.masked_fill(~gate, -torch.inf).topk(3, dim=1).values[:, 2]
    deficit = F.relu(third[eligible] - center_score[eligible] - margin)
    return deficit.square().mean(), int(eligible.sum()), float(deficit.detach().mean())


def fit(data, source, static, saved_state, gate, arm, seed, max_days=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpatialNovelCNN()
    model.load_state_dict(saved_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.02)
    flat = data["candidate_flat"]
    days = []
    for day in range(TRAIN_START, VALID_START):
        if np.any((data["reports"][day].ravel()[flat] > 0) & source):
            days.append(day)
    if max_days is not None:
        days = list(rng.choice(days, size=min(max_days, len(days)), replace=False))
    records = []
    used = 0
    for epoch in range(EPOCHS):
        model.train()
        for day in rng.permutation(days):
            image, _, simple, novel = day_maps(data, int(day), static)
            truth = data["reports"][day].ravel()[flat] > 0
            actual = np.flatnonzero(novel & truth & source)
            known = np.flatnonzero(~novel & truth & source)
            unknown = np.flatnonzero(novel & ~truth & source)
            if not len(unknown):
                continue
            a = rng.choice(actual, size=min(16, len(actual)), replace=False)
            s = rng.choice(known, size=min(8, len(known)), replace=False)
            pos = np.r_[a, s]
            if not len(pos):
                continue
            hard_count = min(len(unknown), max(50, int(len(unknown)*.20)))
            hard_pool = unknown[np.argpartition(simple[unknown],
                                      len(unknown)-hard_count)[-hard_count:]]
            cmp_random = rng.choice(unknown, size=len(pos), replace=True)
            cmp_hard = rng.choice(hard_pool, size=len(pos), replace=True)
            indices = np.r_[np.repeat(pos, 2), np.column_stack((cmp_random, cmp_hard)).ravel()]
            batch = patches(image, flat, indices)
            if len(s):
                batch[2*len(a):2*len(pos), :4, RADIUS, RADIUS] = 0
            if torch.any(batch[:, :4, RADIUS, RADIUS] != 0):
                raise ValueError("Target's own history remains visible")
            if MASK_RATE:
                keep = (torch.rand(len(batch), 1, 2*RADIUS+1, 2*RADIUS+1) >= MASK_RATE).float()
                batch[:, :4] *= keep
            mask_random_blocks(batch)
            local = model(batch)
            scores = local[:, RADIUS, RADIUS]
            pairwise = F.softplus(scores[2*len(pos):] - scores[:2*len(pos)])
            weights = torch.cat((torch.ones(2*len(a)), torch.full((2*len(s),), .5)))
            pair_loss = (pairwise*weights).sum()/weights.sum()
            if not len(a):
                pair_loss = .5*pair_loss
            aux, eligible, mean_deficit = continuity_loss(local, torch.from_numpy(gate[indices]))
            loss = pair_loss + .0005*scores.square().mean() + (LAMBDA*aux if arm != "no_aux" else 0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            used += 1
            records.append((float(pair_loss.detach()), float(aux.detach()), eligible, mean_deficit))
        print(arm, "epoch", epoch+1, "steps", used, flush=True)
    model.eval()
    record = np.array(records)
    return model, {"steps": used, "mean_pairwise_loss": round(float(record[:, 0].mean()), 6),
                   "mean_aux_loss": round(float(record[:, 1].mean()), 6),
                   "mean_eligible_patches_per_batch": round(float(record[:, 2].mean()), 3),
                   "mean_positive_deficit": round(float(record[:, 3].mean()), 6)}


def slim(row):
    return {k: row[k] for k in ("positive_cell_days", "top_10pct_capture",
                               "novel_positive_cell_days", "novel_capture",
                               "daily_positive", "daily_captured", "daily_novel_positive",
                               "daily_novel_captured")}


def pool(folds, arm):
    rows = [f["q4"][arm] for f in folds]
    result = {"positive_cell_days": sum(r["positive_cell_days"] for r in rows),
              "novel_positive_cell_days": sum(r["novel_positive_cell_days"] for r in rows)}
    for field in ("daily_positive", "daily_captured", "daily_novel_positive", "daily_novel_captured"):
        result[field] = np.sum([r[field] for r in rows], axis=0).round(5).tolist()
    result["top_10pct_capture"] = round(sum(result["daily_captured"])/result["positive_cell_days"], 5)
    result["novel_capture"] = round(sum(result["daily_novel_captured"])/result["novel_positive_cell_days"], 5)
    return result


def interval(a, b, novel=True, seed=226, draws=2000):
    positives = np.asarray(a["daily_novel_positive" if novel else "daily_positive"])
    x = np.asarray(a["daily_novel_captured" if novel else "daily_captured"])
    y = np.asarray(b["daily_novel_captured" if novel else "daily_captured"])
    blocks = [np.arange(i, min(i+7, len(positives))) for i in range(0, len(positives), 7)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        ix = np.concatenate([blocks[k] for k in rng.integers(len(blocks), size=len(blocks))])
        if positives[ix].sum():
            out.append((x[ix].sum()-y[ix].sum())/positives[ix].sum())
    return np.quantile(out, [.025, .975]).round(5).tolist()


def self_check():
    logits = torch.zeros((2, 15, 15), requires_grad=True)
    with torch.no_grad():
        logits[0, 7, 7] = -2
        logits[0, 6, 7] = logits[0, 8, 7] = logits[0, 7, 6] = 2
        logits[1, 7, 7] = 2
        logits[1, 6, 7] = logits[1, 8, 7] = logits[1, 7, 6] = -2
    gate = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    loss, eligible, _ = continuity_loss(logits, gate)
    if eligible != 2 or not 7.0 < loss.item() < 8.0:
        raise ValueError("Low surrounded center was not penalized")
    loss.backward()
    if logits.grad[0, 7, 7] >= 0 or logits.grad[1, 7, 7] != 0:
        raise ValueError("Continuity gradient has wrong sign")
    loss2, eligible2, _ = continuity_loss(logits.detach(), gate & torch.tensor([[1, 1, 0, 0]]))
    if eligible2 or loss2.item() != 0:
        raise ValueError("Two neighbors should not activate the rule")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-days", type=int, default=None, help="diagnostic subset only")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    self_check()
    data = load_grid()
    z, _, _, audit = terrain(data)
    ordinary, physical, gate_audit = gates(data, z)
    source_states = torch.load(OUT_MODEL, weights_only=True)["states"]
    district = data["district_id"].ravel()[data["candidate_flat"]]
    splits = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    folds, fitted_states = [], {}
    for number, ids in enumerate(splits[:args.folds], 1):
        target = np.isin(district, ids)
        source = ~target
        static, _ = static_image(data, source)
        saved = source_states[number]["synthetic_center_block"]
        models = {"original": SpatialNovelCNN().eval()}
        models["original"].load_state_dict(saved)
        fit_audit, validation, q4 = {}, {}, {}
        fitted_states[number] = {}
        for arm in ARMS:
            gate = ordinary if arm != "terrain" else physical
            print("fold", number, arm, flush=True)
            model, fit_audit[arm] = fit(data, source, static, saved, gate, arm,
                                         SEED+number*1000, args.max_days)
            models[arm] = model
            fitted_states[number][arm] = {k:v.cpu() for k,v in model.state_dict().items()}
        for arm, model in models.items():
            validation[arm] = slim(q20_rows(data, collect(data, model, static,
                                      range(VALID_START, TEST_START)), source, False)["model"])
            q4[arm] = slim(q20_rows(data, collect(data, model, static,
                                      range(TEST_START, N_DAYS)), target, False)["model"])
        chosen = max(models, key=lambda arm: (validation[arm]["novel_capture"], -list(models).index(arm)))
        folds.append({"fold": number, "training": fit_audit,
                      "source_validation": {k: {"novel_capture":v["novel_capture"],
                                                 "overall_capture":v["top_10pct_capture"]}
                                            for k,v in validation.items()},
                      "source_selected_arm": chosen, "q4": q4})
        print("fold", number, "selected", chosen,
              "q4 novel", {k:v["novel_capture"] for k,v in q4.items()}, flush=True)
    pooled = {arm: pool(folds, arm) for arm in ("original", *ARMS)}
    pooled["source_selected"] = pool([{**f, "q4": {"selected": f["q4"][f["source_selected_arm"]]}}
                                      for f in folds], "selected")
    if args.folds == 5 and args.max_days is None:
        if pooled["original"]["positive_cell_days"] != 7209 or pooled["original"]["novel_positive_cell_days"] != 1664 or pooled["original"]["novel_capture"] != .13529:
            raise ValueError("Original Q4 queue changed")
    output = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "exploratory": args.max_days is not None or args.folds != 5,
              "training_period": "2023-2024", "source_validation": "Jan-Sep 2025",
              "heldout": "Q4 2025, excluded district folds", "lambda": LAMBDA,
              "score_margin": MARGIN, "max_mean_height_gap_m": MAX_GAP_M,
              "learning_rate": LR, "epochs": EPOCHS,
              "max_days": args.max_days, "gate_audit": gate_audit,
              "terrain_source": audit.get("dem", {}).get("source", "cached BMA DTM"),
              "folds": folds, "pooled_q4": pooled,
              "novel_interval_terrain_minus_no_aux": interval(pooled["terrain"], pooled["no_aux"]),
              "novel_interval_terrain_minus_ordinary": interval(pooled["terrain"], pooled["ordinary"]),
              "novel_interval_selected_minus_original": interval(pooled["source_selected"], pooled["original"]),
              "limitations": ["Report-site logits are not flood probabilities or depths",
                              "Unreported cells are not observed dry",
                              "250 m mean DTM cannot confirm an open edge or water surface height",
                              "No complete barrier and event-time drainage state",
                              "Q4 was examined before this experiment"]}
    if args.folds == 5 and args.max_days is None:
        OUTPUT.write_text(json.dumps(output, indent=2)+"\n")
        torch.save({"states": fitted_states}, STATES)
        print("saved", OUTPUT, STATES, flush=True)
    else:
        diagnostic = OUTPUT.with_name("q4_2025_mask_cnn_continuity_loss_diagnostic.json")
        diagnostic.write_text(json.dumps(output, indent=2)+"\n")
        print("saved diagnostic", diagnostic, flush=True)
    print("pooled", {k:(v["novel_capture"], v["top_10pct_capture"]) for k,v in pooled.items()}, flush=True)


if __name__ == "__main__":
    main()
