"""Matched frozen-BMA-CNN test of the 2021 BMA pipe network.

Train a small static CNN correction on source-district 2023-24 Traffy groups.
The existing BMA-road-pretrained, history-masked CNN stays fixed. A shuffled
outlet placement receives the same architecture, training and queue selection.
Neither arm uses report history in the new drainage channels.
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

from flood_citywide_drainage_network import AUDIT, FEATURES
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import ALPHAS, ContextResidualCNN, context, pool
from flood_citywide_isolated_group_mask import (
    BAG_TEMPERATURE, mask_self_check, mask_target_history, sample_group_batch,
)
from flood_citywide_isolated_site_ranker import isolated, queue
from flood_citywide_route_pretrain import lidar_static, terrain
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_bma_drainage_adapter_fivefold.json"
OUT_MODEL = OUT / "q4_2025_bma_drainage_adapter_fivefold.pt"
BMA_JSON = OUT / "q4_2025_bma_road_pretrain_mask_fivefold.json"
BMA_MODEL = OUT / "q4_2025_bma_road_pretrain_mask_fivefold.pt"
EPOCHS = 2
ADAPTER_ALPHAS = (0., .1, .25, .5, 1.)


class StaticDrainageCNN(nn.Module):
    """Learn local interactions around graph-derived drainage access."""

    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(7, 8, 3, padding=1)
        self.body = nn.Sequential(nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1),
                                  nn.ReLU())
        self.head = nn.Conv2d(8, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.body(self.entry(x)))[:, 0]

    def patch_scores(self, x):
        return self.forward(x)[:, RADIUS, RADIUS]

    def full_scores(self, x, candidate):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x[None]))[0].numpy()[candidate]


def adapter_image(data, static, drainage, arm):
    # The first three channels are identical across real and control arms.
    # For LiDAR static: relative elevation, canal distance, built cover.
    pipe = drainage[0]
    outlet = drainage[1 if arm == "real" else 2]
    return np.stack((static[1], static[4], static[7], pipe, outlet,
                     drainage[3], drainage[4])).astype(np.float32)


def fit(data, source, static, static_input, frozen, base_alpha, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    source_grid = np.zeros(SHAPE, bool)
    source_grid.ravel()[data["candidate_flat"]] = source
    flat_to_index = np.full(np.prod(SHAPE), -1, np.int32)
    flat_to_index[data["candidate_flat"]] = np.arange(len(data["candidate_flat"]))
    days = [d for d in range(TRAIN_START, VALID_START)
            if np.any((data["reports"][d] > 0) & isolated(data, d) & source_grid)]
    model = StaticDrainageCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    losses = []
    sampled_groups = 0
    for epoch in range(EPOCHS):
        epoch_losses = []
        model.train()
        for day in rng.permutation(days):
            day = int(day)
            image, _, simple = context(data, day, static)
            groups = sample_group_batch(data, day, source_grid, flat_to_index,
                                        simple, rng, rng, "isolated_groups")
            if not groups:
                continue
            positive = np.concatenate([g[0] for g in groups])
            comparison = np.concatenate([g[1] for g in groups])
            indices = np.r_[positive, comparison]
            history_batch, _ = mask_target_history(
                patches(image, data["candidate_flat"], indices))
            if torch.any(history_batch[:, :4, RADIUS-1:RADIUS+2,
                                              RADIUS-1:RADIUS+2] != 0):
                raise ValueError("Target-local report history was not removed")
            with torch.no_grad():
                fixed = (.6*history_batch[:, 15, RADIUS, RADIUS] +
                         .3*history_batch[:, 16, RADIUS, RADIUS] +
                         .1*history_batch[:, 17, RADIUS, RADIUS])
                base = fixed + base_alpha*frozen.patch_scores(history_batch)
            static_batch = patches(static_input, data["candidate_flat"], indices)
            delta = model.patch_scores(static_batch)
            score = base + delta
            start = 0
            terms = []
            for j, (members, _, _, _) in enumerate(groups):
                n = len(members)
                site = BAG_TEMPERATURE*(
                    torch.logsumexp(score[start:start+n]/BAG_TEMPERATURE, 0)
                    - math.log(n))
                start += n
                negative = score[len(positive)+2*j:len(positive)+2*j+2]
                terms.append(F.softplus(negative-site).mean())
            loss = torch.stack(terms).mean() + .005*delta.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            sampled_groups += len(groups)
        losses.append(round(float(np.mean(epoch_losses)), 5))
        print("adapter epoch", epoch+1, "loss", losses[-1], flush=True)
    model.eval()
    return model, {"training_days": len(days), "groups_sampled": sampled_groups,
                   "epoch_loss": losses}


def score_days(data, scope, static, frozen, base_alpha, correction, days):
    flat = data["candidate_flat"]
    chosen = np.flatnonzero(scope)
    chosen_flat = flat[chosen]
    rows = {str(a): {"all_positive": 0, "all_hits": 0.,
                     "isolated_positive": 0, "isolated_hits": 0.,
                     "isolated_groups": 0, "group_hits": 0.,
                     "daily_isolated_positive": [], "daily_isolated_hits": []}
            for a in ADAPTER_ALPHAS}
    for day in days:
        image, base, simple = context(data, day, static)
        residual = frozen.full_scores(image, data["candidate"])
        novel = simple + base_alpha*residual
        eligible = isolated(data, day)[data["candidate"]][chosen]
        truth = (data["reports"][day].ravel()[flat] > 0)[chosen]
        positive = truth & eligible
        group_grid = np.zeros(SHAPE, bool)
        group_grid.ravel()[chosen_flat] = positive
        components, n_groups = label(group_grid, structure=np.ones((3, 3), bool))
        ids = components.ravel()[chosen_flat]
        for alpha in ADAPTER_ALPHAS:
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
    drainage = np.load(FEATURES)["features"]
    if drainage.shape != (5, *SHAPE):
        raise ValueError("Drainage feature grid differs from report grid")
    with BMA_JSON.with_name(BMA_JSON.stem +
        (f"_seed_{args.seed_offset}" if args.seed_offset else "") + ".json").open() as f:
        previous = json.load(f)
    previous_states = torch.load(BMA_MODEL.with_name(BMA_MODEL.stem +
        (f"_seed_{args.seed_offset}" if args.seed_offset else "") + ".pt"),
        map_location="cpu", weights_only=True)["states"]
    elevation, _, _, _ = terrain(data)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(50), 5)
    records, states = [], {}
    for fold_id, heldout in enumerate(folds, 1):
        if args.fold is not None and fold_id != args.fold:
            continue
        source = ~np.isin(district, heldout)
        static, norm = lidar_static(data, source, elevation)
        mask_self_check(data, static)
        old = next(f for f in previous["folds"] if f["fold"] == fold_id)
        if old["heldout_district_codes"] != [data["districts"][j]["code"] for j in heldout]:
            raise ValueError("BMA model fold mismatch")
        base_alpha = float(old["arms"]["bma_real"]["selected_alpha"])
        frozen = ContextResidualCNN()
        frozen.load_state_dict(previous_states[fold_id]["bma_real"])
        frozen.eval()
        rec = {"fold": fold_id,
               "heldout_district_codes": old["heldout_district_codes"],
               "source_normalization": norm, "frozen_bma_alpha": base_alpha,
               "arms": {}}
        states[fold_id] = {}
        for arm in ("real", "shuffled"):
            x = adapter_image(data, static, drainage, arm)
            model, fit_audit = fit(data, source, static, x, frozen, base_alpha,
                                   SEED+fold_id*1000+args.seed_offset)
            correction = model.full_scores(x, data["candidate"])
            valid = score_days(data, source, static, frozen, base_alpha,
                               correction, range(VALID_START, TEST_START))
            alpha = max(ADAPTER_ALPHAS, key=lambda a: (
                valid[str(a)]["isolated_group_capture"],
                valid[str(a)]["isolated_cell_capture"], -a))
            test = score_days(data, ~source, static, frozen, base_alpha,
                              correction, range(TEST_START, N_DAYS))
            rec["arms"][arm] = {"fit": fit_audit, "source_validation": valid,
                                 "selected_alpha": alpha, "heldout_q4": test}
            states[fold_id][arm] = {k: v.cpu() for k, v in model.state_dict().items()}
            print("fold", fold_id, arm, "alpha", alpha,
                  "Q4 cell/group", test[str(alpha)]["isolated_hits"],
                  test[str(alpha)]["group_hits"], flush=True)
        records.append(rec)
    pooled = {}
    for arm in ("real", "shuffled"):
        pooled[arm] = {str(a): pool([f["arms"][arm]["heldout_q4"][str(a)]
                                     for f in records]) for a in ADAPTER_ALPHAS}
        pooled[arm]["source_selected"] = pool([
            f["arms"][arm]["heldout_q4"][str(f["arms"][arm]["selected_alpha"])]
            for f in records])
        if args.fold is None:
            base = pooled[arm]["0.0"]
            expected = previous["pooled_heldout_q4"]["bma_real"]["source_selected"]
            if abs(base["isolated_hits"]-expected["isolated_hits"]) > .01 or \
               abs(base["group_hits"]-expected["group_hits"]) > .01:
                raise ValueError("Frozen BMA benchmark changed")
    result = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "question": "Do mapped BMA pipe-to-canal candidate paths improve a frozen BMA-road report CNN?",
              "target": "Future Traffy report-positive 250m cell-days, not observed flood extent",
              "drainage_audit": json.loads(AUDIT.read_text()),
              "model": "Frozen BMA-road pretrained target-masked CNN plus trained 3-layer static drainage CNN correction",
              "adapter_features": ["source-normalized relative LiDAR elevation",
                                   "source-normalized OSM canal distance",
                                   "source-normalized 2021 built cover",
                                   "BMA pipe proximity", "candidate pipe-network canal access",
                                   "mapped BMA pump-equipment proximity",
                                   "mapped BMA gate proximity"],
              "training": "2023-24 source-district strict isolated Traffy groups; pairwise group loss; no report history in adapter",
              "control": "Same pipe geometry and district count of candidate outlets; randomly relocated among pipe endpoints",
              "selection": "Correction strength selected on Jan-Sep 2025 source districts only",
              "queue": "Same 10% daily budget and 20% strict-isolated reservation as BMA model",
              "seed_offset": args.seed_offset, "folds": records,
              "pooled_heldout_q4": pooled,
              "limitations": ["Candidate pipe endpoint near named canal is not verified outfall",
                              "Endpoint snaps and cell-to-pipe proximity are uncertain",
                              "No pump/gate operation or true flow direction",
                              "Unreported cells are unknown, not verified dry",
                              "Q4 2025 has been inspected repeatedly; results exploratory"]}
    suffix = (f"_seed_{args.seed_offset}" if args.seed_offset else "") + \
             (f"_fold_{args.fold}" if args.fold else "")
    path = OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    torch.save({"states": states}, OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt"))
    print("pooled", {a: (r["source_selected"]["isolated_hits"],
                         r["source_selected"]["group_hits"])
                     for a, r in pooled.items()}, flush=True)


if __name__ == "__main__":
    main()
