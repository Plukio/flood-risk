"""Matched-path contrastive spatial pretraining for isolated report ranking.

A target receives two otherwise identical patches: one contains a synthetic
history source 2-6 cells away that reaches it along a LiDAR D8 path; the other
contains an equally distant, elevation-matched source with no route to it.
The CNN must rank the connected patch higher. This synthetic task is not a
verified flood label or a validated Bangkok drainage model.

Run: .venv/bin/python flood_citywide_path_contrast_pretrain.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_isolated_site_ranker import day_inputs, isolated, samples
from flood_citywide_route_pretrain import (
    OUT, lidar_static, pooled, score_days, terrain,
)
from flood_citywide_spatial_novel_queue import SpatialNovelCNN


OUT_JSON = OUT / "q4_2025_path_contrast_pretrain_isolated_fivefold.json"
OUT_MODEL = OUT / "q4_2025_path_contrast_pretrain_isolated_fivefold.pt"
PAIR_CACHE = OUT / "path_contrast_d8_pairs_250m.npz"
PATH_MIN_HOPS = 2
PATH_MAX_HOPS = 6
MAX_ELEVATION_GAP_M = .5
PRETRAIN_AMPLITUDE = float(np.log1p(1.))
REPORT_EPOCHS = 2


def _offsets_by_distance():
    result = {}
    for dr in range(-RADIUS, RADIUS+1):
        for dc in range(-RADIUS, RADIUS+1):
            d2 = dr*dr+dc*dc
            if d2 >= 4:
                result.setdefault(d2, []).append((dr, dc))
    return result


def build_pairs(data: dict, z: np.ndarray, node_flat: np.ndarray,
                down: np.ndarray):
    """One positive/negative source pair per D8 path, with distance matched."""
    if PAIR_CACHE.exists():
        with np.load(PAIR_CACHE) as saved:
            if not np.array_equal(saved["candidate_flat"], data["candidate_flat"]):
                raise ValueError("Cached contrastive pairs differ from grid")
            return saved["pairs"], json.loads(str(saved["audit"].item()))
    h, w = SHAPE
    lookup = np.full(h*w, -1, np.int32)
    lookup[node_flat] = np.arange(len(node_flat), dtype=np.int32)
    node_to_candidate = np.searchsorted(data["candidate_flat"], node_flat)
    if not np.array_equal(data["candidate_flat"][node_to_candidate], node_flat):
        raise ValueError("Terrain node differs from candidate grid")
    connected = [set() for _ in range(len(node_flat))]
    paths = []
    for source in range(len(node_flat)):
        target = source
        sr, sc = divmod(int(node_flat[source]), w)
        for hop in range(1, 21):
            target = int(down[target])
            if target < 0:
                break
            connected[target].add(source)
            if not PATH_MIN_HOPS <= hop <= PATH_MAX_HOPS:
                continue
            tr, tc = divmod(int(node_flat[target]), w)
            dr, dc = sr-tr, sc-tc
            if 4 <= dr*dr+dc*dc <= RADIUS*RADIUS*2 and (
                    abs(dr) <= RADIUS and abs(dc) <= RADIUS):
                paths.append((source, target, dr, dc))
    options = _offsets_by_distance()
    candidate_z = z.ravel()[node_flat]
    chosen = []
    gaps = []
    for source, target, pdr, pdc in paths:
        tr, tc = divmod(int(node_flat[target]), w)
        best, best_gap = None, float("inf")
        for ndr, ndc in options[pdr*pdr+pdc*pdc]:
            nr, nc = tr+ndr, tc+ndc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            negative = int(lookup[nr*w+nc])
            if negative < 0 or negative in connected[target]:
                continue
            # Match the source elevation, preventing a trivial uphill cue.
            gap = abs(float(candidate_z[negative]-candidate_z[source]))
            if gap < best_gap:
                best, best_gap = (negative, ndr, ndc), gap
        if best is None or best_gap > MAX_ELEVATION_GAP_M:
            continue
        negative, ndr, ndc = best
        chosen.append((node_to_candidate[target], source, negative,
                       pdr, pdc, ndr, ndc))
        gaps.append(best_gap)
    pairs = np.asarray(chosen, np.int32).reshape(-1, 7)
    if len(pairs) < 1000:
        raise ValueError(f"Too few matched D8 path pairs: {len(pairs)}")
    gaps = np.asarray(gaps)
    audit = {"candidate_path_pairs": len(paths),
             "matched_pairs": len(pairs),
             "max_elevation_gap_m": MAX_ELEVATION_GAP_M,
             "source_elevation_gap_median": round(float(np.median(gaps)), 4),
             "source_elevation_gap_p90": round(float(np.percentile(gaps, 90)), 4),
             "positive_hops": [PATH_MIN_HOPS, PATH_MAX_HOPS],
             "source_center_distance": "exact same squared Euclidean distance",
             "negative_rule": "source does not reach target within 20 D8 hops"}
    PAIR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PAIR_CACHE, pairs=pairs,
                        candidate_flat=data["candidate_flat"],
                        audit=json.dumps(audit))
    return pairs, audit


def pair_batch(image: np.ndarray, candidate_flat: np.ndarray,
               pairs: np.ndarray, rng: np.random.Generator):
    """Same target, distance and distractors; only route connectivity varies."""
    n = len(pairs)
    targets = np.repeat(pairs[:, 0], 2)
    batch = patches(image, candidate_flat, targets)
    arange = torch.arange(n)
    positive_row = torch.from_numpy(pairs[:, 3]+RADIUS)
    positive_col = torch.from_numpy(pairs[:, 4]+RADIUS)
    negative_row = torch.from_numpy(pairs[:, 5]+RADIUS)
    negative_col = torch.from_numpy(pairs[:, 6]+RADIUS)
    batch[2*arange, 0, positive_row, positive_col] = PRETRAIN_AMPLITUDE
    batch[2*arange+1, 0, negative_row, negative_col] = PRETRAIN_AMPLITUDE
    # A few common report-like distractors make this more than one-hot routing.
    for j in range(n):
        for _ in range(int(rng.integers(0, 4))):
            rr, cc = rng.integers(0, 2*RADIUS+1, size=2)
            if abs(int(rr)-RADIUS) <= 1 and abs(int(cc)-RADIUS) <= 1:
                continue
            if ((rr == int(positive_row[j]) and cc == int(positive_col[j])) or
                (rr == int(negative_row[j]) and cc == int(negative_col[j]))):
                continue
            if batch[2*j, -2, rr, cc] == 0:
                continue
            batch[2*j:2*j+2, 0, rr, cc] = PRETRAIN_AMPLITUDE
    if torch.any(batch[:, :4, RADIUS-1:RADIUS+2,
                              RADIUS-1:RADIUS+2] != 0):
        raise ValueError("Contrastive pretraining exposes immediate history")
    return batch


def pretrain(data: dict, source: np.ndarray, static: np.ndarray,
             pairs: np.ndarray, node_flat: np.ndarray, seed: int,
             steps: int, batch_size: int):
    rng = np.random.default_rng(seed+491)
    torch.manual_seed(seed)
    model = SpatialNovelCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    image = np.concatenate((np.zeros((4, *SHAPE), np.float32), static,
                            data["candidate"][None].astype(np.float32),
                            np.ones((1, *SHAPE), np.float32)))
    node_scope = source[np.searchsorted(data["candidate_flat"], node_flat)]
    training = pairs[source[pairs[:, 0]] &
                     node_scope[pairs[:, 1]] & node_scope[pairs[:, 2]]]
    heldout = pairs[(~source[pairs[:, 0]]) &
                    (~node_scope[pairs[:, 1]]) & (~node_scope[pairs[:, 2]])]
    if len(training) < 1000 or len(heldout) < 100:
        raise ValueError(f"Insufficient geographic path pairs {len(training)}, {len(heldout)}")
    losses = []
    model.train()
    for step in range(steps):
        chosen = training[rng.choice(len(training), batch_size, replace=True)]
        batch = pair_batch(image, data["candidate_flat"], chosen, rng)
        scores = model.patch_scores(batch).reshape(-1, 2)
        loss = F.softplus(scores[:, 1]-scores[:, 0]).mean()
        loss = loss + .0005*scores.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 100 == 0:
            print("path step", step+1, "loss", round(np.mean(losses[-100:]), 5),
                  flush=True)
    model.eval()
    accuracy, no_terrain_accuracy, margins = [], [], []
    with torch.no_grad():
        for part in np.array_split(heldout[rng.choice(
                len(heldout), min(2000, len(heldout)), replace=False)],
                max(1, int(np.ceil(min(2000, len(heldout))/batch_size)))):
            batch = pair_batch(image, data["candidate_flat"], part, rng)
            scores = model.patch_scores(batch).reshape(-1, 2).numpy()
            no_terrain = batch.clone()
            no_terrain[:, 4:7] = 0
            ablated = model.patch_scores(no_terrain).reshape(-1, 2).numpy()
            margin = scores[:, 0]-scores[:, 1]
            margins.extend(margin.tolist())
            accuracy.extend((margin > 0).tolist())
            no_terrain_accuracy.extend((ablated[:, 0] > ablated[:, 1]).tolist())
    fit = {"steps": steps,
           "source_matched_pairs": len(training),
           "excluded_matched_pairs": len(heldout),
           "loss_first_50": round(float(np.mean(losses[:50])), 5),
           "loss_last_50": round(float(np.mean(losses[-50:])), 5),
           "heldout_synthetic_pair_accuracy": round(float(np.mean(accuracy)), 5),
           "heldout_no_lidar_pair_accuracy": round(float(np.mean(
               no_terrain_accuracy)), 5),
           "heldout_synthetic_pair_mean_margin": round(float(np.mean(margins)), 5),
           "heldout_synthetic_test_pairs": len(accuracy)}
    return model, fit


def finetune(data: dict, source: np.ndarray, static: np.ndarray,
             seed: int, pretrained: SpatialNovelCNN, transfer_mode: str):
    """Map synthetic source to 365-day history; optionally reset the head."""
    torch.manual_seed(seed)
    model = SpatialNovelCNN()
    if transfer_mode == "full":
        model.load_state_dict(pretrained.state_dict())
    elif transfer_mode == "reset_head":
        model.entry.load_state_dict(pretrained.entry.state_dict())
        model.steps.load_state_dict(pretrained.steps.state_dict())
    else:
        raise ValueError(f"Unknown transfer mode: {transfer_mode}")
    with torch.no_grad():
        model.entry.weight[:, 1:4].zero_()
    rng = np.random.default_rng(seed)
    scope = np.zeros(SHAPE, bool)
    scope.ravel()[data["candidate_flat"]] = source
    days = [d for d in range(TRAIN_START, VALID_START)
            if np.any((data["reports"][d] > 0) & isolated(data, d) & scope)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    losses, n_groups = [], 0
    for epoch in range(REPORT_EPOCHS):
        model.train()
        epoch_losses = []
        for day in rng.permutation(days):
            image, _, simple, _ = day_inputs(data, int(day), static)
            picked = samples(data, int(day), scope, simple, rng, limit=12)
            if picked is None:
                continue
            positive, comparison, groups = picked
            n_groups += groups
            batch = patches(image, data["candidate_flat"],
                            np.r_[positive, comparison])
            if torch.any(batch[:, :4, RADIUS-1:RADIUS+2,
                                      RADIUS-1:RADIUS+2] != 0):
                raise ValueError("Isolated report history exposed")
            score = model.patch_scores(batch)
            loss = F.softplus(score[len(positive):]-score[:len(positive)]).mean()
            loss = loss + .0005*score.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(round(float(np.mean(epoch_losses)), 5))
        print("report epoch", epoch+1, "loss", losses[-1], flush=True)
    model.eval()
    return model, {"train_days": len(days), "groups_across_epochs": n_groups,
                   "epoch_loss": losses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-steps", type=int, default=600)
    parser.add_argument("--pretrain-batch", type=int, default=128)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--transfer-mode", choices=("full", "reset_head"),
                        default="full")
    args = parser.parse_args()
    out_json = (OUT_JSON if args.transfer_mode == "full" else
                OUT_JSON.with_name(OUT_JSON.stem+"_reset_head.json"))
    out_model = (OUT_MODEL if args.transfer_mode == "full" else
                 OUT_MODEL.with_name(OUT_MODEL.stem+"_reset_head.pt"))
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_grid()
    elevation, node_flat, down, terrain_audit = terrain(data)
    pairs, pair_audit = build_pairs(data, elevation, node_flat, down)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    records, states = [], {}
    for fold_id, ids in enumerate(folds, 1):
        source = ~np.isin(district, ids)
        static, norm = lidar_static(data, source, elevation)
        seed = SEED + fold_id*1000 + args.seed_offset
        print("fold", fold_id, "source", int(source.sum()), flush=True)
        pre, synthetic_fit = pretrain(data, source, static, pairs,
                                      node_flat, seed,
                                      args.pretrain_steps, args.pretrain_batch)
        print("synthetic", fold_id, synthetic_fit, flush=True)
        tuned, report_fit = finetune(data, source, static, seed, pre,
                                     args.transfer_mode)
        models = {"path_contrast_pretrained_cnn": tuned}
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
        states[fold_id] = {
            "synthetic": {k: v.cpu() for k, v in pre.state_dict().items()},
            "report": {k: v.cpu() for k, v in tuned.state_dict().items()}}
        print("fold", fold_id, {name: (r["isolated_hits"],
                                     r["group_hits"]) for name, r in
                                     test.items()}, flush=True)
    result = {"built_utc": datetime.now(timezone.utc).isoformat(
                  timespec="seconds"),
              "question": "Does matched connected-vs-unconnected path pretraining improve emerging report-site ranking?",
              "synthetic_target": "Rank a source connected by 2-6 D8 hops above an equally distant source with <=0.5m elevation mismatch and no route to the target",
              "synthetic_loss": "pairwise softplus(disconnected score - connected score) plus 0.0005 score-square penalty",
              "transfer": args.transfer_mode,
              "transfer_note": "retain learned source entry and spatial trunk; source input becomes 365-day report history; zero other history weights; optionally reset output head",
              "terrain": terrain_audit,
              "pair_audit": pair_audit,
              "pretrain_steps": args.pretrain_steps,
              "pretrain_batch": args.pretrain_batch,
              "seed_offset": args.seed_offset,
              "training_dates": ["2023-01-01", "2024-12-31"],
              "source_validation_dates": ["2025-01-01", "2025-09-30"],
              "geographic_test_dates": ["2025-10-01", "2025-12-31"],
              "queue": "10% per heldout fold per day; 20% reserve for strictly isolated candidates",
              "folds": records,
              "pooled_source_validation": pooled(records, "source_validation"),
              "pooled_heldout_q4": pooled(records, "heldout_q4"),
              "limitations": ["D8 connectivity is a terrain proxy, not observed drainage or flooding",
                              "Synthetic source is mapped to prior report history at fine-tuning",
                              "Unreported report cells are unknown, not verified dry",
                              "Cached LiDAR is present-day rather than vintage-matched",
                              "Q4 2025 has been inspected repeatedly; this is exploratory"]}
    if (result["pooled_heldout_q4"]["distant_history"]["isolated_positive"] != 332 or
        result["pooled_heldout_q4"]["distant_history"]["isolated_hits"] != 65.):
        raise ValueError("Frozen isolated queue changed")
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"states": states}, out_model)
    print("pooled", {k: (v["isolated_hits"], v["group_hits"],
                         v["all_capture"]) for k, v in
                         result["pooled_heldout_q4"].items()}, flush=True)
    print("saved", out_json, out_model, flush=True)


if __name__ == "__main__":
    main()
