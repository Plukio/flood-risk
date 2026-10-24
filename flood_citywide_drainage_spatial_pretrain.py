"""Pretrain on matched pipe-component relations using all mapped surface layers.

Target: at a fixed receiver cell, rank a synthetic prior source on the same
snapped BMA pipe component above an equally distant source on a different
component. Source elevation and built cover are matched. This teaches a mapped
relation, not observed flow or flooding. A frozen BMA-road CNN supplies the
existing report score; the pretrained trunk learns a report-rank correction.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import label
from torch import nn
from torch.nn import functional as F

from flood_citywide_drainage_adapter import BMA_JSON, BMA_MODEL
from flood_citywide_drainage_network import AUDIT as NETWORK_AUDIT, FEATURES
from flood_citywide_drainage_spatial_pairs import AUDIT as PAIR_AUDIT, build as build_pairs
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import ContextResidualCNN, context, pool
from flood_citywide_isolated_group_mask import (
    BAG_TEMPERATURE, mask_self_check, mask_target_history, sample_group_batch,
)
from flood_citywide_isolated_site_ranker import isolated, queue
from flood_citywide_route_pretrain import lidar_static, terrain
from flood_citywide_spatial_novel_queue import SpatialNovelCNN
from flood_citywide_surface_osm import AUDIT as BUILDING_AUDIT, build as buildings
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_bma_drainage_spatial_pretrain_fivefold.json"
OUT_MODEL = OUT / "q4_2025_bma_drainage_spatial_pretrain_fivefold.pt"
PRETRAIN_STEPS = 500
PRETRAIN_BATCH = 96
REPORT_EPOCHS = 2
ALPHAS = (0., .1, .25, .5, 1.)
SOURCE_AMPLITUDE = float(np.log1p(1))
ARMS = ("scratch", "connected_pretrained", "shuffled_pretrained")


class SurfaceContextCNN(SpatialNovelCNN):
    """Seven-cell radius; source, terrain, roads, cover, buildings, drains."""

    def __init__(self):
        super().__init__()
        # 0 synthetic source / prior-365 reports; 1:11 LiDAR, OSM and cover;
        # 11 candidate; 12:16 BMA pipe/outlet/pump/gate; 16 OSM buildings;
        # 17 valid grid mask.
        self.entry = nn.Conv2d(18, 8, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


def surface_image(data, static, drainage, building):
    image = np.concatenate((
        np.zeros((1, *SHAPE), np.float32), static,
        data["candidate"][None].astype(np.float32),
        drainage[[0, 1, 3, 4]], building[None].astype(np.float32),
        np.ones((1, *SHAPE), np.float32)))
    if image.shape != (18, *SHAPE) or not np.isfinite(image).all():
        raise ValueError("Invalid all-spatial pretraining image")
    return image


def pair_batch(image, candidate_flat, pairs, rng):
    """Two views of the same receiver with one synthetic source moved."""
    n = len(pairs)
    batch = patches(image, candidate_flat, np.repeat(pairs[:, 0], 2))
    arange = torch.arange(n)
    pr = torch.from_numpy(pairs[:, 3]+RADIUS)
    pc = torch.from_numpy(pairs[:, 4]+RADIUS)
    nr = torch.from_numpy(pairs[:, 5]+RADIUS)
    nc = torch.from_numpy(pairs[:, 6]+RADIUS)
    batch[2*arange, 0, pr, pc] = SOURCE_AMPLITUDE
    batch[2*arange+1, 0, nr, nc] = SOURCE_AMPLITUDE
    # Identical distractors in both views make the relation conditional on
    # which source moved, while keeping report-like sparse context.
    for j in range(n):
        for _ in range(int(rng.integers(0, 4))):
            row, col = map(int, rng.integers(0, 2*RADIUS+1, size=2))
            if abs(row-RADIUS) <= 1 and abs(col-RADIUS) <= 1:
                continue
            if ((row == int(pr[j]) and col == int(pc[j])) or
                (row == int(nr[j]) and col == int(nc[j])) or
                batch[2*j, 11, row, col] == 0):
                continue
            batch[2*j:2*j+2, 0, row, col] = SOURCE_AMPLITUDE
    if torch.any(batch[:, 0, RADIUS-1:RADIUS+2, RADIUS-1:RADIUS+2]):
        raise ValueError("Synthetic source enters the masked receiver neighborhood")
    return batch


def split_pairs(pairs, source):
    train = source[pairs[:, 0]] & source[pairs[:, 1]] & source[pairs[:, 2]]
    test = ~source[pairs[:, 0]] & ~source[pairs[:, 1]] & ~source[pairs[:, 2]]
    return np.flatnonzero(train), np.flatnonzero(test)


def synthetic_accuracy(model, image, candidate_flat, heldout, seed):
    rng = np.random.default_rng(seed+472)
    selected = heldout[rng.choice(len(heldout), min(1500, len(heldout)), replace=False)]
    masks = {"all": (), "no_bma_gis": (12, 16),
             "no_lidar": (1, 4), "no_roads_canals": (4, 8),
             "no_worldcover": (8, 11), "no_buildings": (16, 17)}
    correct = {key: [] for key in masks}
    model.eval()
    with torch.no_grad():
        for part in np.array_split(selected, max(1, math.ceil(len(selected)/PRETRAIN_BATCH))):
            batch = pair_batch(image, candidate_flat, part, rng)
            for name, (start, stop) in masks.items():
                altered = batch if name == "all" else batch.clone()
                if name != "all":
                    altered[:, start:stop] = 0
                scores = model.patch_scores(altered).reshape(-1, 2)
                correct[name].extend((scores[:, 0] > scores[:, 1]).tolist())
    return {name: round(float(np.mean(values)), 5)
            for name, values in correct.items()}


def pretrain(data, source, image, pairs, seed, shuffled):
    train_idx, heldout_idx = split_pairs(pairs, source)
    if len(train_idx) < 1000 or len(heldout_idx) < 100:
        raise ValueError(f"Insufficient geographically split network pairs: {len(train_idx)}, {len(heldout_idx)}")
    rng = np.random.default_rng(seed+491)
    flip_rng = np.random.default_rng(seed+95531)
    torch.manual_seed(seed)
    model = SurfaceContextCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    flips = np.zeros(len(pairs), bool)
    if shuffled:
        flips[train_idx[flip_rng.choice(len(train_idx), len(train_idx)//2,
                                        replace=False)]] = True
    losses = []
    model.train()
    for step in range(PRETRAIN_STEPS):
        chosen_idx = train_idx[rng.choice(len(train_idx), PRETRAIN_BATCH, replace=True)]
        batch = pair_batch(image, data["candidate_flat"], pairs[chosen_idx], rng)
        scores = model.patch_scores(batch).reshape(-1, 2)
        sign = torch.from_numpy(np.where(flips[chosen_idx], -1., 1.).astype(np.float32))
        margin = sign*(scores[:, 0]-scores[:, 1])
        loss = F.softplus(-margin).mean()+.0005*scores.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 100 == 0:
            print("synthetic", "shuffled" if shuffled else "connected",
                  "step", step+1, "loss", round(float(np.mean(losses[-100:])), 5),
                  flush=True)
    quality = synthetic_accuracy(model, image, data["candidate_flat"],
                                 pairs[heldout_idx], seed)
    return model, {"source_pairs": len(train_idx),
                   "heldout_pairs": len(heldout_idx),
                   "steps": PRETRAIN_STEPS,
                   "loss_first_50": round(float(np.mean(losses[:50])), 5),
                   "loss_last_50": round(float(np.mean(losses[-50:])), 5),
                   "heldout_pair_accuracy": quality,
                   "shuffled_half_labels": shuffled}


def initialize(seed, pretrained, model_factory=SurfaceContextCNN):
    torch.manual_seed(seed)
    model = model_factory()
    if pretrained is not None:
        model.entry.load_state_dict(pretrained.entry.state_dict())
        model.steps.load_state_dict(pretrained.steps.state_dict())
        # Reset synthetic relation head before fitting report correction.
        if torch.count_nonzero(model.head.weight):
            raise ValueError("Report head was not reset")
    return model


def fit_reports(data, source, static, template, frozen, base_alpha, seed,
                pretrained, model_factory=SurfaceContextCNN):
    model = initialize(seed, pretrained, model_factory)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.02)
    rng = np.random.default_rng(seed)
    source_grid = np.zeros(SHAPE, bool)
    source_grid.ravel()[data["candidate_flat"]] = source
    flat_to_index = np.full(np.prod(SHAPE), -1, np.int32)
    flat_to_index[data["candidate_flat"]] = np.arange(len(data["candidate_flat"]))
    days = [d for d in range(TRAIN_START, VALID_START)
            if np.any((data["reports"][d] > 0) & isolated(data, d) & source_grid)]
    losses, n_groups = [], 0
    for epoch in range(REPORT_EPOCHS):
        model.train()
        epoch_losses = []
        for day in rng.permutation(days):
            history_image, _, simple = context(data, int(day), static)
            groups = sample_group_batch(data, int(day), source_grid,
                                        flat_to_index, simple, rng, rng,
                                        "isolated_groups")
            if not groups:
                continue
            positive = np.concatenate([g[0] for g in groups])
            comparison = np.concatenate([g[1] for g in groups])
            indices = np.r_[positive, comparison]
            masked, _ = mask_target_history(
                patches(history_image, data["candidate_flat"], indices))
            with torch.no_grad():
                fixed = (.6*masked[:, 15, RADIUS, RADIUS] +
                         .3*masked[:, 16, RADIUS, RADIUS] +
                         .1*masked[:, 17, RADIUS, RADIUS])
                base = fixed+base_alpha*frozen.patch_scores(masked)
            image = template.copy()
            image[0] = history_image[0]
            batch = patches(image, data["candidate_flat"], indices)
            if torch.any(batch[:, 0, RADIUS-1:RADIUS+2, RADIUS-1:RADIUS+2]):
                raise ValueError("Report source leaks into target history mask")
            delta = model.patch_scores(batch)
            score = base+delta
            start = 0
            terms = []
            for j, (members, _, _, _) in enumerate(groups):
                count = len(members)
                site = BAG_TEMPERATURE*(
                    torch.logsumexp(score[start:start+count]/BAG_TEMPERATURE, 0)
                    -math.log(count))
                start += count
                negative = score[len(positive)+2*j:len(positive)+2*j+2]
                terms.append(F.softplus(negative-site).mean())
            loss = torch.stack(terms).mean()+.005*delta.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            n_groups += len(groups)
        losses.append(round(float(np.mean(epoch_losses)), 5))
        print("report epoch", epoch+1, "loss", losses[-1], flush=True)
    model.eval()
    return model, {"training_days": len(days), "groups_sampled": n_groups,
                   "epoch_loss": losses}


def score_days(data, scope, static, template, frozen, base_alpha, model, days):
    flat = data["candidate_flat"]
    chosen = np.flatnonzero(scope)
    chosen_flat = flat[chosen]
    rows = {str(a): {"all_positive": 0, "all_hits": 0.,
                     "isolated_positive": 0, "isolated_hits": 0.,
                     "isolated_groups": 0, "group_hits": 0.,
                     "daily_isolated_positive": [], "daily_isolated_hits": []}
            for a in ALPHAS}
    for day in days:
        history_image, base, simple = context(data, day, static)
        main_score = frozen.full_scores(history_image, data["candidate"])
        image = template.copy()
        image[0] = history_image[0]
        correction = model.full_scores(image, data["candidate"])
        novel = simple+base_alpha*main_score
        eligible = isolated(data, day)[data["candidate"]][chosen]
        truth = (data["reports"][day].ravel()[flat] > 0)[chosen]
        positive = truth & eligible
        grid = np.zeros(SHAPE, bool)
        grid.ravel()[chosen_flat] = positive
        components, n_groups = label(grid, structure=np.ones((3, 3), bool))
        ids = components.ravel()[chosen_flat]
        for alpha in ALPHAS:
            weights = queue(base[chosen], novel[chosen]+alpha*correction[chosen], eligible)
            row = rows[str(alpha)]
            row["all_positive"] += int(truth.sum())
            row["all_hits"] += float(weights[truth].sum())
            row["isolated_positive"] += int(positive.sum())
            hits = float(weights[positive].sum())
            row["isolated_hits"] += hits
            row["isolated_groups"] += n_groups
            row["group_hits"] += sum(float(weights[ids == j].max())
                                     for j in range(1, n_groups+1))
            row["daily_isolated_positive"].append(int(positive.sum()))
            row["daily_isolated_hits"].append(round(hits, 5))
    for row in rows.values():
        row["all_capture"] = round(row["all_hits"]/row["all_positive"], 5)
        row["isolated_cell_capture"] = round(
            row["isolated_hits"]/row["isolated_positive"], 5)
        row["isolated_group_capture"] = round(
            row["group_hits"]/row["isolated_groups"], 5)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = load_grid()
    pairs, pair_audit = build_pairs(data)
    drainage = np.load(FEATURES)["features"]
    building = buildings()
    elevation, _, _, _ = terrain(data)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(50), 5)
    suffix = f"_seed_{args.seed_offset}" if args.seed_offset else ""
    previous = json.loads(BMA_JSON.with_name(BMA_JSON.stem+suffix+".json").read_text())
    previous_states = torch.load(BMA_MODEL.with_name(BMA_MODEL.stem+suffix+".pt"),
                                 map_location="cpu", weights_only=True)["states"]
    records, states = [], {}
    for fold_id, heldout in enumerate(folds, 1):
        if args.fold is not None and fold_id != args.fold:
            continue
        source = ~np.isin(district, heldout)
        static, normalization = lidar_static(data, source, elevation)
        template = surface_image(data, static, drainage, building)
        mask_self_check(data, static)
        seed = SEED+fold_id*1000+args.seed_offset
        old = next(f for f in previous["folds"] if f["fold"] == fold_id)
        if old["heldout_district_codes"] != [data["districts"][j]["code"]
                                              for j in heldout]:
            raise ValueError("Saved BMA-road fold differs")
        base_alpha = float(old["arms"]["bma_real"]["selected_alpha"])
        frozen = ContextResidualCNN()
        frozen.load_state_dict(previous_states[fold_id]["bma_real"])
        frozen.eval()
        print("fold", fold_id, "source", int(source.sum()), flush=True)
        real_pre, real_audit = pretrain(data, source, template, pairs, seed, False)
        shuffled_pre, shuffled_audit = pretrain(data, source, template, pairs,
                                               seed, True)
        record = {"fold": fold_id,
                  "heldout_district_codes": old["heldout_district_codes"],
                  "normalization": normalization, "frozen_bma_alpha": base_alpha,
                  "pretraining": {"connected": real_audit,
                                  "shuffled": shuffled_audit}, "arms": {}}
        states[fold_id] = {}
        for arm, initial in (("scratch", None),
                             ("connected_pretrained", real_pre),
                             ("shuffled_pretrained", shuffled_pre)):
            model, fit_audit = fit_reports(data, source, static, template,
                                           frozen, base_alpha, seed, initial)
            valid = score_days(data, source, static, template, frozen, base_alpha,
                               model, range(VALID_START, TEST_START))
            alpha = max(ALPHAS, key=lambda a: (
                valid[str(a)]["isolated_group_capture"],
                valid[str(a)]["isolated_cell_capture"], -a))
            test = score_days(data, ~source, static, template, frozen, base_alpha,
                              model, range(TEST_START, N_DAYS))
            record["arms"][arm] = {"fit": fit_audit,
                                   "source_validation": valid,
                                   "selected_alpha": alpha,
                                   "heldout_q4": test}
            states[fold_id][arm] = {k: v.cpu() for k, v in model.state_dict().items()}
            print("fold", fold_id, arm, "alpha", alpha, "Q4 cells/groups",
                  test[str(alpha)]["isolated_hits"],
                  test[str(alpha)]["group_hits"], flush=True)
        records.append(record)
    pooled = {}
    for arm in ARMS:
        pooled[arm] = {str(a): pool([f["arms"][arm]["heldout_q4"][str(a)]
                                     for f in records]) for a in ALPHAS}
        pooled[arm]["source_selected"] = pool([
            f["arms"][arm]["heldout_q4"][str(f["arms"][arm]["selected_alpha"])]
            for f in records])
        if args.fold is None:
            expected = previous["pooled_heldout_q4"]["bma_real"]["source_selected"]
            base = pooled[arm]["0.0"]
            if (abs(base["isolated_hits"]-expected["isolated_hits"]) > .01 or
                abs(base["group_hits"]-expected["group_hits"]) > .01 or
                base["isolated_positive"] != 332 or base["isolated_groups"] != 158):
                raise ValueError("Frozen BMA-road comparator changed")
    result = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "question": "Does paired pipe-connectivity pretraining improve emerging report-site transfer when all mapped spatial context is available?",
              "target": "Traffy report-positive 250m cell-days, not verified flood extent",
              "features": ["source/prior-365 report map", "BMA LiDAR elevation, relative elevation, slope",
                           "OSM canal length/distance and road length/distance",
                           "WorldCover built, crop, tree/grass cover",
                           "BMA pipe proximity, candidate canal path, pump/gate proximity",
                           "2025 OSM building footprint fraction", "candidate/valid masks"],
              "synthetic_target": "Same receiver, equal map distance: source on same 5m-snapped pipe component ranks above source on different component; source elevation and built fraction matched",
              "synthetic_loss": "softplus(score(disconnected)-score(connected)) plus output regularization",
              "control": "Half of source-pair labels flipped before pretraining, fixed across steps; same input and updates",
              "transfer": "Copy entry and five spatial steps; reset synthetic head; fine-tune all spatial weights with target-masked report group loss",
              "report_training": "2023-24 source districts, strict isolated groups against unreported comparisons; frozen BMA-road CNN as base",
              "selection": "Jan-Sep 2025 source districts only; maximize group then cell capture, then smaller correction",
              "queue": "10% candidate cells daily; 20% of slots reserved for strict isolated cells",
              "pair_audit": pair_audit,
              "network_audit": json.loads(NETWORK_AUDIT.read_text()),
              "building_audit": json.loads(BUILDING_AUDIT.read_text()),
              "seed_offset": args.seed_offset, "folds": records,
              "pooled_heldout_q4": pooled,
              "limitations": ["Same pipe component is inferred, not verified hydraulic flow",
                              "Synthetic source transfer does not prove flood prediction",
                              "No observed pump/gate state, outlet head or verified inlet",
                              "Unreported comparisons are unknown, not verified dry",
                              "Q4 2025 has been repeatedly inspected; exploratory"]}
    suffix += f"_fold_{args.fold}" if args.fold else ""
    OUT_JSON.with_name(OUT_JSON.stem+suffix+".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    torch.save({"states": states}, OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt"))
    print("pooled", {arm: (row["source_selected"]["isolated_hits"],
                            row["source_selected"]["group_hits"])
                     for arm, row in pooled.items()}, flush=True)


if __name__ == "__main__":
    main()
