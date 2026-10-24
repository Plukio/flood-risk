"""Mask augmentation ablation for the Bangkok zero-own-history spatial CNN.

Compare the existing 10% random-neighbor mask model with extra synthetic
center-history removal, and with additional random 3x3 report-history holes.
Choose any recipe using source-district validation novelty capture only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_spatial_novel_queue import (
    MASK_RATE, OUT_JSON as PRIOR_JSON, OUT_MODEL as PRIOR_MODEL,
    SpatialNovelCNN, aggregate, collect, day_maps, evaluate, static_image,
)
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval

OUT_JSON = OUT / "q4_2025_spatial_mask_augmentation_fivefold.json"
OUT_MODEL = OUT / "q4_2025_spatial_mask_augmentation_fivefold.pt"
ARMS = ("current_random10", "synthetic_center", "synthetic_center_block")
QUOTA = .20  # 20% of the fixed 10%-of-cells inspection slots.
EPOCHS = 2


def mask_random_blocks(batch: torch.Tensor, probability: float = .5) -> int:
    """Hide random 3x3 historical-report blocks within candidate patches."""
    chosen = torch.rand(len(batch)) < probability
    rows = torch.randint(0, 2*RADIUS-1, (len(batch),))
    cols = torch.randint(0, 2*RADIUS-1, (len(batch),))
    for i in torch.nonzero(chosen).flatten().tolist():
        r, c = int(rows[i]), int(cols[i])
        batch[i, :4, r:r+3, c:c+3] = 0
    return int(chosen.sum())


def fit(data: dict, source: np.ndarray, static: np.ndarray, seed: int,
        block_mask: bool, hard_fraction: float = .20,
        hard_floor: int = 50):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpatialNovelCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    flat = data["candidate_flat"]
    days = []
    actual_count, synthetic_count = 0, 0
    for day in range(TRAIN_START, VALID_START):
        own365 = data["prefix"][day].ravel()[flat] == \
                 data["prefix"][day-365].ravel()[flat]
        truth = data["reports"][day].ravel()[flat] > 0
        actual_count += int((own365 & truth & source).sum())
        synthetic_count += int((~own365 & truth & source).sum())
        if np.any(truth & source):
            days.append(day)
    print("training days",len(days),"actual novel",actual_count,
          "eligible synthetic known",synthetic_count,flush=True)
    losses_by_epoch, used_actual, used_synthetic, blocked = [], 0, 0, 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        losses = []
        for day in rng.permutation(days):
            image, _, simple, novel = day_maps(data, int(day), static)
            truth = data["reports"][day].ravel()[flat] > 0
            actual = np.flatnonzero(novel & truth & source)
            known = np.flatnonzero(~novel & truth & source)
            unknown = np.flatnonzero(novel & ~truth & source)
            if not len(unknown):
                continue
            a = rng.choice(actual, size=min(16,len(actual)), replace=False)
            s = rng.choice(known, size=min(8,len(known)), replace=False)
            pos = np.r_[a,s]
            if not len(pos):
                continue
            hard_count = min(len(unknown), max(hard_floor,
                                               int(len(unknown)*hard_fraction)))
            hard_pool = unknown[np.argpartition(simple[unknown],
                                      len(unknown)-hard_count)[-hard_count:]]
            random_cmp = rng.choice(unknown,size=len(pos),replace=True)
            hard_cmp = rng.choice(hard_pool,size=len(pos),replace=True)
            positive_indices = np.repeat(pos,2)
            comparison_indices = np.column_stack((random_cmp,hard_cmp)).ravel()
            batch = patches(image,flat,np.r_[positive_indices,comparison_indices])
            if len(s):
                # Known repeat sites become synthetic cold-start examples;
                # no scored-day outcome or neighboring history is altered.
                batch[2*len(a):2*len(pos),:4,RADIUS,RADIUS] = 0
            if torch.any(batch[:,:4,RADIUS,RADIUS] != 0):
                raise ValueError("Own report history remains visible")
            if MASK_RATE:
                keep = (torch.rand(len(batch),1,2*RADIUS+1,
                                   2*RADIUS+1) >= MASK_RATE).float()
                batch[:,:4] *= keep
            if block_mask:
                blocked += mask_random_blocks(batch)
            scores = model.patch_scores(batch)
            per_pair = F.softplus(scores[len(pos)*2:]-scores[:len(pos)*2])
            weights = torch.cat((torch.ones(2*len(a)),
                                 torch.full((2*len(s),),.5)))
            loss = (per_pair*weights).sum()/weights.sum()
            if not len(a):
                loss = .5*loss
            loss = loss+.0005*scores.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            used_actual += len(a)
            used_synthetic += len(s)
        losses_by_epoch.append(round(float(np.mean(losses)),5))
        print("epoch",epoch,"loss",losses_by_epoch[-1],flush=True)
    model.eval()
    return model, {"epoch_pairwise_loss":losses_by_epoch,
                   "source_actual_novel_positive_rows":actual_count,
                   "source_known_positive_rows_eligible_to_mask":synthetic_count,
                   "actual_positive_samples_used":used_actual,
                   "synthetic_positive_samples_used":used_synthetic,
                   "random_neighbor_mask_rate":MASK_RATE,
                   "hard_negative_pool_fraction":hard_fraction,
                   "hard_negative_pool_floor":hard_floor,
                   "extra_3x3_block_probability":.5 if block_mask else 0,
                   "training_patches_with_extra_block":blocked}


def q20(data, model, static, days, scope, cluster):
    rows = evaluate(data,collect(data,model,static,days),scope,cluster)
    return rows["spatial_cnn_quota_20pct"],rows["simple_neighbor_quota_20pct"],rows["fixed_full"]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    data = load_grid()
    old = json.loads(PRIOR_JSON.read_text())
    old_states = torch.load(PRIOR_MODEL,weights_only=True)["states"]
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds,states = [],{}
    district_folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])),5)
    for fold_number,ids in enumerate(district_folds,1):
        target = np.isin(district,ids)
        source = ~target
        static,normalization = static_image(data,source)
        models = {}
        current = SpatialNovelCNN()
        current.load_state_dict(old_states[fold_number])
        current.eval()
        models[ARMS[0]] = current
        fits = {ARMS[0]:{"source":"frozen earlier 10% random-neighbor-mask model"}}
        states[fold_number] = {}
        for arm,block in ((ARMS[1],False),(ARMS[2],True)):
            print("fold",fold_number,arm,flush=True)
            models[arm],fits[arm] = fit(data,source,static,
                                        SEED+fold_number*1000,block)
            states[fold_number][arm] = {k:v.cpu() for k,v in
                                        models[arm].state_dict().items()}
        validation = {}
        for arm,model in models.items():
            row,_,_ = q20(data,model,static,range(VALID_START,TEST_START),
                          source,False)
            validation[arm] = {"overall_capture":row["top_10pct_capture"],
                               "first_in_year_capture":row["novel_capture"]}
        chosen = max(ARMS,key=lambda arm:validation[arm]["first_in_year_capture"])
        q4 = {}
        for arm,model in models.items():
            row,simple,fixed = q20(data,model,static,
                                   range(TEST_START,N_DAYS),target,True)
            q4[arm] = row
            if "simple_neighbor" not in q4:
                q4["simple_neighbor"] = simple
                q4["fixed_full"] = fixed
        prior = old["folds"][fold_number-1]["q4"]["spatial_cnn_quota_20pct"]
        if q4[ARMS[0]]["top_10pct_capture"] != prior["top_10pct_capture"]:
            raise ValueError("Frozen original CNN did not reproduce")
        folds.append({"fold":fold_number,
                      "district_codes":[data["districts"][j]["code"] for j in ids],
                      "normalization":normalization,"fit":fits,
                      "source_validation":validation,
                      "source_selected_arm":chosen,"q4":q4})
        print("fold",fold_number,"source choice",chosen,
              {k:(v["top_10pct_capture"],v["novel_capture"])
               for k,v in q4.items()},flush=True)
    names = ["fixed_full","simple_neighbor",*ARMS]
    pooled = {name:aggregate(folds,name) for name in names}
    if (pooled["fixed_full"]["positive_cell_days"] != 7209 or
        pooled["fixed_full"]["novel_positive_cell_days"] != 1664 or
        pooled["fixed_full"]["top_10pct_capture"] != .47113 or
        pooled["simple_neighbor"]["top_10pct_capture"] != .44114 or
        pooled["current_random10"]["top_10pct_capture"] != .43639):
        raise ValueError("Matched novelty queue changed")
    selected_rows = [f["q4"][f["source_selected_arm"]] for f in folds]
    positives = np.sum([r["daily_positive"] for r in selected_rows],axis=0)
    hits = np.sum([r["daily_captured"] for r in selected_rows],axis=0)
    novel_positive = sum(r["novel_positive_cell_days"] for r in selected_rows)
    novel_hits = sum(sum(r["daily_novel_captured"]) for r in selected_rows)
    clusters = sum(r["same_day_adjacent_clusters"] for r in selected_rows)
    cluster_hits = sum(r["cluster_capture"]*r["same_day_adjacent_clusters"]
                       for r in selected_rows)
    pooled["source_selected_mask_arm"] = {
        "positive_cell_days":int(positives.sum()),
        "top_10pct_capture":round(float(hits.sum()/positives.sum()),5),
        "novel_positive_cell_days":novel_positive,
        "novel_capture":round(novel_hits/novel_positive,5),
        "same_day_adjacent_clusters":clusters,
        "cluster_capture":round(cluster_hits/clusters,5),
        "daily_positive":positives.astype(int).tolist(),
        "daily_captured":np.round(hits,5).tolist()}
    pooled["selected_minus_simple_weekly_interval"] = paired_interval(
        pooled["source_selected_mask_arm"],pooled["simple_neighbor"])
    output = {"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "target":"Q4 report-positive zero-own-365d cells in same daily 10% queues; not verified flood",
              "fixed_novel_slot_share_of_queue":QUOTA,
              "arm_selection":"source Jan-Sep 2025 first-in-a-year capture at 20% reserved inspection slots",
              "training_period":["2023-01-01","2024-12-31"],
              "validation_period":["2025-01-01","2025-09-30"],
              "heldout_test_period":["2025-10-01","2025-12-31"],
              "arms":ARMS,"folds":folds,"pooled_fold_queues":pooled,
              "limitations":["Synthetic masked repeat sites may differ from real first-in-a-year sites",
                             "Random masks do not establish physical water connectivity",
                             "Unreported cells are unknown, not verified dry",
                             "Q4 2025 has been inspected repeatedly in earlier work",
                             "Static physical snapshots may differ from event dates"]}
    OUT_JSON.write_text(json.dumps(output,indent=2)+"\n")
    torch.save({"states":states},OUT_MODEL)
    print("pooled",{k:(pooled[k]["top_10pct_capture"],pooled[k]["novel_capture"])
                    for k in names+["source_selected_mask_arm"]},flush=True)
    print("saved",OUT_JSON,OUT_MODEL,flush=True)


if __name__ == "__main__":
    main()
