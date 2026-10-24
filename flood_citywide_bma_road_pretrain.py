"""Weak BMA road-corridor pretraining, then matched target-masked Traffy fit.

The BMA label means a flood was recorded somewhere on a named road in a
district. It does not label every cell in that road bag as wet. Unreported
comparison roads have unknown flood status.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from flood_citywide_bma_road_bags import AUDIT_JSON, load_bags
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import ALPHAS, pool, score_days
from flood_citywide_isolated_group_mask import mask_self_check, train
from flood_citywide_path_pretrain_mask_transfer import (
    ARM, OUT, PRETRAIN_MODEL, OUT_JSON as CONTROL_JSON,
)
from flood_citywide_route_pretrain import lidar_static, terrain
from flood_citywide_spatial_novel_queue import SpatialNovelCNN

OUT_JSON = OUT / "q4_2025_bma_road_pretrain_mask_fivefold.json"
OUT_MODEL = OUT / "q4_2025_bma_road_pretrain_mask_fivefold.pt"
MIN_CELLS, MAX_CELLS = 2, 30
TEMP = .05
EPOCHS = 3


def bag_score(scores: torch.Tensor) -> torch.Tensor:
    return TEMP*(torch.logsumexp(scores/TEMP, 0)-math.log(len(scores)))


def bag_pairs(bags, district, records, reported, source_ids, seed):
    """One unique reported road and a size-matched unreported road per pair."""
    rng = np.random.default_rng(seed)
    size = np.asarray([len(b) for b in bags])
    positives = sorted(set(map(int, records)))
    controls = np.flatnonzero(~reported & (size >= MIN_CELLS) &
                                  (size <= MAX_CELLS) &
                                  np.isin(district, list(source_ids)))
    pairs = []
    counts = {"positive_road_bags": 0, "no_matched_control": 0,
              "overlapping_control_excluded": 0}
    for p in positives:
        if district[p] not in source_ids or not MIN_CELLS <= size[p] <= MAX_CELLS:
            continue
        counts["positive_road_bags"] += 1
        options = controls[(district[controls] == district[p]) &
                           (size[controls] >= max(MIN_CELLS, size[p]//2)) &
                           (size[controls] <= min(MAX_CELLS, 2*size[p]))]
        if not len(options):
            counts["no_matched_control"] += 1
            continue
        # Prefer similar-size roads that do not simply cross the same cells.
        order = rng.permutation(len(options))
        options = options[order]
        options = sorted(options, key=lambda n: abs(math.log(size[n]/size[p])))
        viable = [int(n) for n in options if len(np.intersect1d(
            bags[p], bags[n]))/min(size[p], size[n]) <= .25]
        if not viable:
            counts["overlapping_control_excluded"] += 1
            continue
        pairs.append((p, viable[0]))
    counts["usable_pairs"] = len(pairs)
    counts["unique_source_districts"] = len({int(district[p]) for p, _ in pairs})
    return pairs, counts


def static_pretrain_image(data, static):
    """History channels are zero, including the synthetic path source."""
    image = np.zeros((16, *SHAPE), np.float32)
    image[4:14] = static
    image[14] = data["candidate"].astype(np.float32)
    image[15] = 1.
    return image


def pretrain(data, static, starting_state, bags, pairs, seed, shuffled):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpatialNovelCNN()
    model.load_state_dict(starting_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.02)
    image = static_pretrain_image(data, static)
    flat = data["candidate_flat"]
    # Assign half the size-matched pairs the opposite label once, then keep
    # the assignment fixed across epochs. This destroys the expected link
    # between BMA reporting and the spatial input while preserving bag sizes.
    flip = np.zeros(len(pairs), bool)
    if shuffled:
        flip[rng.choice(len(pairs), len(pairs)//2, replace=False)] = True
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for j in rng.permutation(len(pairs)):
            actual, other = pairs[int(j)]
            positive, comparison = ((other, actual) if flip[int(j)] else
                                    (actual, other))
            selected = np.r_[bags[positive], bags[comparison]]
            batch = patches(image, flat, selected)
            if torch.count_nonzero(batch[:, :4]):
                raise ValueError("BMA pretraining received report-history input")
            scores = model.patch_scores(batch)
            n = len(bags[positive])
            margin = bag_score(scores[:n])-bag_score(scores[n:])
            loss = F.softplus(-margin)+.0005*scores.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(round(float(np.mean(epoch_losses)), 5))
        print("BMA", "shuffled" if shuffled else "real", "epoch", epoch+1,
              "loss", losses[-1], flush=True)
    model.eval()
    return model, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = load_grid()
    bags, bag_district, record_day, record_bag, _, reported = load_bags(data)
    if any(record_day >= VALID_START):
        raise ValueError("BMA pretraining includes a validation or test event")
    elevation, _, _, terrain_audit = terrain(data)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    saved = torch.load(PRETRAIN_MODEL, map_location="cpu", weights_only=True)
    suffix = "" if args.seed_offset == 0 else f"_seed_{args.seed_offset}"
    if args.fold is not None:
        suffix += f"_fold_{args.fold}"
    control_path = CONTROL_JSON.with_name(CONTROL_JSON.stem +
        ("" if args.seed_offset == 0 else f"_seed_{args.seed_offset}") + ".json")
    with control_path.open() as file:
        matched_control = json.load(file)
    records, states = [], {}
    for fold_id, heldout in enumerate(folds, 1):
        if args.fold is not None and fold_id != args.fold:
            continue
        source = ~np.isin(district, heldout)
        source_ids = set(range(len(data["districts"]))) - set(map(int, heldout))
        static, norm = lidar_static(data, source, elevation)
        mask_self_check(data, static)
        pair_list, pair_audit = bag_pairs(bags, bag_district, record_bag,
                                          reported, source_ids,
                                          SEED + fold_id*1000)
        if len(pair_list) < 25:
            raise ValueError("Too few source-district road-bag pairs")
        print("fold", fold_id, "pairs", len(pair_list), pair_audit, flush=True)
        control = next(f for f in matched_control["folds"] if f["fold"] == fold_id)
        if control["heldout_district_codes"] != [data["districts"][j]["code"]
                                                  for j in heldout]:
            raise ValueError("Saved matched control uses different geographic fold")
        rec = {"fold": fold_id,
               "heldout_district_codes": control["heldout_district_codes"],
               "normalization": norm, "road_pair_audit": pair_audit,
               "saved_path_control": control["arms"]["path_pretrained"],
               "arms": {}}
        states[fold_id] = {}
        for name, shuffled in (("bma_real", False), ("bma_shuffled", True)):
            pre, pre_loss = pretrain(data, static,
                saved["states"][fold_id]["synthetic"], bags, pair_list,
                SEED+fold_id*1000+args.seed_offset, shuffled)
            model, fit = train(data, source, static,
                               SEED+fold_id*1000+args.seed_offset, ARM,
                               pretrained=pre)
            valid = score_days(data, source, static, model,
                               range(VALID_START, TEST_START))
            alpha = max(ALPHAS, key=lambda a: (
                valid[str(a)]["isolated_group_capture"],
                valid[str(a)]["isolated_cell_capture"], -a))
            test = score_days(data, ~source, static, model,
                              range(TEST_START, N_DAYS))
            rec["arms"][name] = {"pretrain_loss": pre_loss,
                                  "fit": fit, "source_validation": valid,
                                  "selected_alpha": alpha, "heldout_q4": test}
            states[fold_id][name] = {k: v.cpu() for k, v in model.state_dict().items()}
            print("fold", fold_id, name, "alpha", alpha,
                  "q4 cells", test[str(alpha)]["isolated_hits"],
                  "groups", test[str(alpha)]["group_hits"], flush=True)
        records.append(rec)
    pooled = {}
    for name in ("bma_real", "bma_shuffled"):
        pooled[name] = {
            str(a): pool([f["arms"][name]["heldout_q4"][str(a)]
                          for f in records]) for a in ALPHAS}
        pooled[name]["source_selected"] = pool([
            f["arms"][name]["heldout_q4"][str(f["arms"][name]["selected_alpha"])]
            for f in records])
        if args.fold is None:
            baseline = pooled[name]["0.0"]
            if (baseline["isolated_positive"] != 332 or
                baseline["isolated_groups"] != 158 or
                baseline["isolated_hits"] != 65. or
                baseline["group_hits"] != 44.):
                raise ValueError("Frozen strict-isolated queue changed")
    result = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": "Does weak BMA road-corridor supervision improve path-pretrained, target-masked CNN transfer?",
        "target": "Traffy report-positive 250m cell-days, not verified wet extent",
        "bma_road_audit": json.loads(AUDIT_JSON.read_text()),
        "bag_filter_cells": [MIN_CELLS, MAX_CELLS],
        "bag_score": "0.05 * logmeanexp(member CNN scores / 0.05)",
        "bma_loss": "softplus(comparison road-bag score - reported road-bag score); each unique road once per epoch",
        "bma_inputs": "Source-normalized LiDAR, road, canal, WorldCover, candidate and grid-valid channels; all four history channels zero",
        "shuffled_control": "Randomly flip the report label in half of same-district, similar-size road pairs, fixed across epochs; same pairs and update count",
        "bma_training_years": [2022, 2023, 2024],
        "report_training_arm": ARM,
        "path_pretrain_checkpoint": str(PRETRAIN_MODEL),
        "matched_control_json": str(control_path),
        "fine_tuning_seed_offset": args.seed_offset,
        "terrain": terrain_audit,
        "folds": records, "pooled_heldout_q4": pooled,
        "limitations": ["Road bag is an uncertain corridor, not an exact flood coordinate",
                        "Unreported road and cell comparisons are unknown, not verified dry",
                        "BMA records cover selected roads; OSM 2025 geometry is not event-vintage matched",
                        "Q4 2025 has been inspected repeatedly; this is exploratory",
                        "No verified flood extent or rainfall scenario validity"],
    }
    out_json = OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    out_model = OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    torch.save({"states": states}, out_model)
    print("pooled", {n: (p["source_selected"]["isolated_hits"],
                        p["source_selected"]["group_hits"])
                     for n, p in pooled.items()}, flush=True)
    print("saved", out_json, out_model, flush=True)


if __name__ == "__main__":
    main()
