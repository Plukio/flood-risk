"""Continue a frozen-recipe masked CNN with a differentiable novel queue loss.

Both continuation arms start from the same saved block-masked CNN. They use
the same actual first-in-a-year source positives, days, optimizer and epochs.
One keeps the pairwise objective; the other maximizes positive inclusion in a
smooth 20%-of-inspection-slots novel queue on the full eligible source grid.
Unreported cells are queue competitors, not asserted dry observations.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from flood_citywide_history_ranker import N_DAYS, TEST_START, TRAIN_START, VALID_START, load_grid
from flood_citywide_history_cnn_transfer import SEED
from flood_citywide_mask_cnn_blend import OUT_JSON as CONTROL_JSON, score_days
from flood_citywide_spatial_mask_augmentation import OUT_MODEL as CONTROL_MODEL
from flood_citywide_spatial_novel_queue import SpatialNovelCNN, collect, day_maps, static_image
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_mask_cnn_soft_queue_fivefold.json"
OUT_MODEL = OUT / "q4_2025_mask_cnn_soft_queue_fivefold.pt"
ARMS = ("pairwise_continue", "soft_queue_continue")
EPOCHS = 2
LR = .0002
TAU = .10
NOVEL_RESERVE = .20 * .10  # fraction of all candidate cells


def soft_capacity_loss(scores: torch.Tensor, positive_mask: torch.Tensor,
                       capacity: float, tau: float = TAU) -> torch.Tensor:
    """Positive log inclusion at a differentiable fixed-capacity threshold.

    Bisection finds the detached root, then one Newton correction supplies the
    implicit threshold gradient. Its derivative routes a positive's update
    through the other cells near the same day's queue boundary.
    """
    if not 0 < capacity < scores.numel() or not positive_mask.any():
        raise ValueError("Invalid queue training day")
    with torch.no_grad():
        detached = scores.detach()
        lo = detached.min() - 20*tau
        hi = detached.max() + 20*tau
        for _ in range(35):
            mid = (lo+hi)/2
            if torch.sigmoid((detached-mid)/tau).sum() > capacity:
                lo = mid
            else:
                hi = mid
        root = (lo+hi)/2
    p0 = torch.sigmoid((scores-root)/tau)
    slope = (p0*(1-p0)).sum().clamp_min(1e-6)
    threshold = root + tau*(p0.sum()-capacity)/slope
    return F.softplus((threshold-scores[positive_mask])/tau).mean()


def loss_check():
    torch.manual_seed(7)
    scores = torch.randn(100, requires_grad=True)
    positive = torch.zeros(100, dtype=torch.bool)
    positive[[8, 40, 75]] = True
    loss = soft_capacity_loss(scores, positive, 3.0)
    loss.backward()
    if not torch.isfinite(scores.grad).all() or scores.grad[positive].mean() >= 0:
        raise ValueError("Soft queue gradients invalid")
    with torch.no_grad():
        for offset in (-2.0, 0.0, 2.0):
            shifted = scores.detach()+offset
            if abs(float(soft_capacity_loss(shifted, positive, 3.0)-loss)) > 1e-4:
                raise ValueError("Soft queue loss should ignore a shared logit shift")


def fit_epoch(model, optimizer, data, source, static, days, seed, arm):
    rng = np.random.default_rng(seed)
    flat = data["candidate_flat"]
    source_count = int(source.sum())
    losses = []
    model.train()
    for day in rng.permutation(days):
        day_rng = np.random.default_rng(seed+int(day)*9973)
        image, _, simple, novel = day_maps(data, int(day), static)
        truth = data["reports"][day].ravel()[flat] > 0
        eligible = np.flatnonzero(novel & source)
        positives = np.flatnonzero(novel & truth & source)
        if not len(positives):
            continue
        # The same sampled true positives and day order are used in both arms.
        sampled = day_rng.choice(positives, size=min(16,len(positives)),replace=False)
        full = model(torch.from_numpy(image[None]))[0].reshape(-1)
        candidate_scores = full[torch.from_numpy(flat)]
        if arm == "soft_queue_continue":
            positions = np.searchsorted(eligible,sampled)
            positive_mask = torch.zeros(len(eligible),dtype=torch.bool)
            positive_mask[positions] = True
            loss = soft_capacity_loss(candidate_scores[eligible],positive_mask,
                                      NOVEL_RESERVE*source_count)
        else:
            unknown = np.flatnonzero(novel & ~truth & source)
            hard_count = min(len(unknown),max(50,len(unknown)//5))
            hard_pool = unknown[np.argpartition(simple[unknown],
                                    len(unknown)-hard_count)[-hard_count:]]
            random_cmp = day_rng.choice(unknown,size=len(sampled),replace=True)
            hard_cmp = day_rng.choice(hard_pool,size=len(sampled),replace=True)
            pos_scores = candidate_scores[torch.from_numpy(sampled)]
            loss = .5*(F.softplus(candidate_scores[torch.from_numpy(random_cmp)]-
                                 pos_scores).mean()+
                       F.softplus(candidate_scores[torch.from_numpy(hard_cmp)]-
                                 pos_scores).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return round(float(np.mean(losses)),5),len(losses)


def choose_source(model, data, source, static):
    return score_days(collect(data,model,static,range(VALID_START,TEST_START)),
                      source,alphas=(1.,))["1.0"]


def q4_row(model, data, target, static):
    return score_days(collect(data,model,static,range(TEST_START,N_DAYS)),
                      target,alphas=(1.,))["1.0"]


def main():
    torch.set_num_threads(4)
    loss_check()
    data = load_grid()
    control = json.loads(CONTROL_JSON.read_text())
    control_states = torch.load(CONTROL_MODEL,weights_only=True)["states"]
    district = data["district_id"].ravel()[data["candidate_flat"]]
    district_folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])),5)
    folds,states = [],{}
    for fold_number,ids in enumerate(district_folds,1):
        source = ~np.isin(district,ids)
        target = ~source
        static,_ = static_image(data,source)
        old_state = control_states[fold_number]["synthetic_center_block"]
        days = [day for day in range(TRAIN_START,VALID_START)
                if np.any((data["reports"][day].ravel()[data["candidate_flat"]]>0)&
                          source & (data["prefix"][day].ravel()[data["candidate_flat"]] ==
                                    data["prefix"][day-365].ravel()[data["candidate_flat"]]))]
        old_fold = control["folds"][fold_number-1]
        source_rows = {"original":old_fold["source_validation"]["1.0"]}
        audit,models = {},{}
        states[fold_number] = {}
        for arm in ARMS:
            model = SpatialNovelCNN()
            model.load_state_dict(old_state)
            optimizer = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.02)
            best = source_rows["original"]["novel_capture"]
            best_epoch = 0
            best_state = copy.deepcopy(old_state)
            epoch_rows = []
            print("fold",fold_number,arm,"training days",len(days),flush=True)
            for epoch in range(1,EPOCHS+1):
                train_loss,used_days = fit_epoch(model,optimizer,data,source,static,
                                                 days,SEED+fold_number*1000+epoch,arm)
                val = choose_source(model,data,source,static)
                epoch_rows.append({"epoch":epoch,"train_loss":train_loss,
                                   "used_days":used_days,"source_novel_capture":
                                   val["novel_capture"]})
                print("fold",fold_number,arm,"epoch",epoch,
                      "source",val["novel_capture"],flush=True)
                if val["novel_capture"]>best+1e-5:
                    best = val["novel_capture"]
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    source_rows[arm] = val
            model.load_state_dict(best_state)
            model.eval()
            models[arm] = model
            states[fold_number][arm] = {k:v.cpu() for k,v in best_state.items()}
            audit[arm] = {"epochs":epoch_rows,"source_selected_epoch":best_epoch,
                          "source_selected_novel_capture":best}
            if best_epoch == 0:
                source_rows[arm] = source_rows["original"]
        chosen = max(("original",*ARMS),key=lambda a:
                     (source_rows[a]["novel_capture"],-(("original",*ARMS).index(a))))
        q4 = {"original":old_fold["q4"]["1.0"]}
        for arm,model in models.items():
            print("fold",fold_number,"scoring Q4",arm,flush=True)
            q4[arm] = q4_row(model,data,target,static)
        folds.append({"fold":fold_number,"train_days":len(days),
                      "source_validation":{k:{"novel_capture":v["novel_capture"],
                                               "overall_capture":v["overall_capture"]}
                                           for k,v in source_rows.items()},
                      "audit":audit,"source_selected_arm":chosen,"q4":q4})
        print("fold",fold_number,"selected",chosen,"Q4 novel",
              {k:v["novel_capture"] for k,v in q4.items()},flush=True)

    def pool(choice):
        rows = [f["q4"][choice(f)] for f in folds]
        positive = sum(r["positive"] for r in rows)
        novel_positive = sum(r["novel_positive"] for r in rows)
        hits = sum(r["captured"] for r in rows)
        novel_hits = sum(r["novel_captured"] for r in rows)
        return {"positive":positive,"novel_positive":novel_positive,
                "captured":round(hits,5),"novel_captured":round(novel_hits,5),
                "overall_capture":round(hits/positive,5),
                "novel_capture":round(novel_hits/novel_positive,5)}
    pooled = {name:pool(lambda f,name=name:name) for name in ("original",*ARMS)}
    pooled["source_selected"] = pool(lambda f:f["source_selected_arm"])
    if (pooled["original"]["novel_positive"]!=1664 or
        pooled["original"]["novel_capture"]!=.13529):
        raise ValueError("Original matched queue changed")
    result = {"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "question":"Does soft fixed-capacity queue training improve the masked CNN over a matched pairwise continuation?",
              "queue":"Q4 2025 five district folds, 10% daily capacity, 20% of slots for first-in-a-year candidates",
              "training":"2023-2024 true first-in-a-year source positives; same dates, initialization, optimizer, epochs and sampled positive cells in continuation arms",
              "selection":"best epoch and arm by Jan-Sep 2025 source novel capture; ties keep older model",
              "learning_rate":LR,"soft_queue_temperature":TAU,
              "folds":folds,"pooled_q4":pooled,
              "limitations":["No synthetic repeat positives in either continuation arm; original baseline did use them",
                             "Unreported cells are unknown rather than observed dry",
                             "Q4 has been inspected repeatedly in this project"]}
    OUT_JSON.write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"states":states},OUT_MODEL)
    print("pooled",pooled,flush=True)
    print("saved",OUT_JSON,OUT_MODEL,flush=True)


if __name__ == "__main__":
    main()
