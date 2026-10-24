"""Group-level version of the 13.53% historical-repeat masking CNN.

Uses the same zero-own-prior-year daily queue and held-out district folds as
the earlier synthetic-center-plus-random-block test. Training compares a
same-day touching positive group with unreported candidates and applies the
target-center mask to every group member and comparison. Repeat-site groups
augment actual first-in-a-year groups at lower weight. Unreported is unknown.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import label, maximum_filter
from torch.nn import functional as F

from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_spatial_novel_queue import (
    SpatialNovelCNN, collect, day_maps, static_image,
)
from flood_citywide_spatial_mask_augmentation import (
    OUT_JSON as PRIOR_JSON, mask_random_blocks,
)
from flood_citywide_target_site_mask_test import aggregate, q20_rows
from flood_citywide_isolated_group_mask import groups_in_mask
from flood_multi_target_citywide_data import OUT

OUT_JSON=OUT / "q4_2025_group_loss_history_mask_fivefold.json"
OUT_MODEL=OUT / "q4_2025_group_loss_history_mask_fivefold.pt"
EPOCHS=2
ACTUAL_LIMIT=16
REPEAT_LIMIT=8
REPEAT_WEIGHT=.5
BAG_TEMPERATURE=.1
PIXEL_HISTORY_DROPOUT=.10


def sample_groups(data:dict,day:int,source_grid:np.ndarray,
                  flat_to_index:np.ndarray,simple:np.ndarray,
                  actual_rng:np.random.Generator,
                  repeat_rng:np.random.Generator):
    truth=data["reports"][day]>0
    known=truth & source_grid
    prior=(data["prefix"][day]-data["prefix"][day-365])>0
    actual=groups_in_mask(known & ~prior,actual_rng,
                          flat_to_index,ACTUAL_LIMIT)
    repeat=groups_in_mask(known & prior,repeat_rng,
                          flat_to_index,REPEAT_LIMIT)
    if not actual and not repeat:
        return None
    # Target districts' scored-day labels do not affect comparison sampling.
    safe=source_grid & ~prior & ~maximum_filter(known.astype(np.uint8),
                                                  size=3,mode="constant").astype(bool)
    unknown=np.flatnonzero(safe.ravel())
    if not len(unknown):
        return None
    pool=flat_to_index[unknown]
    hard_n=min(len(pool),max(50,len(pool)//5))
    hard=pool[np.argpartition(simple[pool],-hard_n)[-hard_n:]]
    groups=[]
    for members,weight,rng,kind in (
        *((m,1.,actual_rng,"actual") for m in actual),
        *((m,REPEAT_WEIGHT,repeat_rng,"repeat") for m in repeat)):
        random=int(rng.choice(pool))
        difficult=int(rng.choice(hard))
        groups.append((members,np.array([random,difficult],np.int32),weight,kind))
    return groups


def fit(data:dict,source:np.ndarray,static:np.ndarray,seed:int):
    torch.manual_seed(seed)
    actual_rng=np.random.default_rng(seed)
    repeat_rng=np.random.default_rng(seed+913_721)
    source_grid=np.zeros(SHAPE,bool)
    source_grid.ravel()[data["candidate_flat"]]=source
    flat_to_index=np.full(np.prod(SHAPE),-1,np.int32)
    flat_to_index[data["candidate_flat"]]=np.arange(len(data["candidate_flat"]))
    days=[d for d in range(TRAIN_START,VALID_START)
          if np.any((data["reports"][d]>0)&source_grid)]
    model=SpatialNovelCNN()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.02)
    stats={"actual_groups":0,"repeat_groups":0,
           "repeat_member_patches_with_masked_own_history":0,
           "random_blocked_patches":0,"epoch_pairwise_loss":[]}
    for epoch in range(EPOCHS):
        model.train()
        losses=[]
        for day in actual_rng.permutation(days):
            image,_,simple,_=day_maps(data,int(day),static)
            groups=sample_groups(data,int(day),source_grid,flat_to_index,
                                 simple,actual_rng,repeat_rng)
            if not groups:
                continue
            positive=np.concatenate([g[0] for g in groups])
            comparison=np.concatenate([g[1] for g in groups])
            batch=patches(image,data["candidate_flat"],
                          np.r_[positive,comparison])
            # Target-cell mask is symmetric for every positive member and
            # comparison. It removes real own history from repeat groups.
            repeat_members=np.concatenate([g[0] for g in groups
                                           if g[3]=="repeat"]) if any(
                g[3]=="repeat" for g in groups) else np.empty(0,np.int32)
            stats["repeat_member_patches_with_masked_own_history"]+=int((
                image[0].ravel()[data["candidate_flat"][repeat_members]]>0).sum())
            batch[:,:4,RADIUS,RADIUS]=0
            if torch.any(batch[:,:4,RADIUS,RADIUS]!=0):
                raise ValueError("Target center report history remains visible")
            keep=(torch.rand(len(batch),1,2*RADIUS+1,
                             2*RADIUS+1)>=PIXEL_HISTORY_DROPOUT).float()
            batch[:,:4]*=keep
            stats["random_blocked_patches"]+=mask_random_blocks(batch)
            scores=model.patch_scores(batch)
            bag_losses=[]
            start=0
            for j,(members,_,weight,kind) in enumerate(groups):
                n=len(members)
                site=(torch.logsumexp(scores[start:start+n]/BAG_TEMPERATURE,
                                      dim=0)-math.log(n))*BAG_TEMPERATURE
                start+=n
                cmp=scores[len(positive)+2*j:len(positive)+2*j+2]
                bag_losses.append(weight*F.softplus(cmp-site).mean())
                stats[kind+"_groups"]+=1
            loss=torch.stack(bag_losses).sum()/sum(g[2] for g in groups)
            loss=loss+.0005*scores.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        stats["epoch_pairwise_loss"].append(round(float(np.mean(losses)),5))
        print("epoch",epoch+1,"loss",stats["epoch_pairwise_loss"][-1],flush=True)
    model.eval()
    stats["train_days_with_source_reports"]=len(days)
    if (stats["actual_groups"]==0 or stats["repeat_groups"]==0 or
        stats["repeat_member_patches_with_masked_own_history"]==0 or
        stats["random_blocked_patches"]==0):
        raise ValueError("Historical masked-group augmentation was inactive")
    return model,stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed-offset",type=int,default=0)
    args=parser.parse_args()
    suffix="" if args.seed_offset==0 else f"_seed_{args.seed_offset}"
    out_json=OUT_JSON if not suffix else OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    out_model=OUT_MODEL if not suffix else OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    data=load_grid()
    prior=json.loads(PRIOR_JSON.read_text())
    district=data["district_id"].ravel()[data["candidate_flat"]]
    folds=np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])),5)
    records,states=[],{}
    for fold_id,ids in enumerate(folds,1):
        source=~np.isin(district,ids)
        static,norm=static_image(data,source)
        print("fold",fold_id,flush=True)
        model,training=fit(data,source,static,
                           SEED+fold_id*1000+args.seed_offset)
        validation=q20_rows(data,collect(data,model,static,
                                          range(VALID_START,TEST_START)),
                            source,False)["model"]
        test=q20_rows(data,collect(data,model,static,
                                    range(TEST_START,N_DAYS)),
                      ~source,True)["model"]
        records.append({"fold":fold_id,
                        "heldout_district_codes":[data["districts"][j]["code"]
                                                  for j in ids],
                        "normalization":norm,"training":training,
                        "source_validation":{"overall_capture":validation[
                            "top_10pct_capture"],"first_in_year_capture":validation[
                            "novel_capture"]},
                        "q4":{"group_loss_history_mask":test}})
        states[fold_id]={k:v.cpu() for k,v in model.state_dict().items()}
        print("fold",fold_id,"q4",test["top_10pct_capture"],
              test["novel_capture"],flush=True)
    pooled=aggregate(records,"group_loss_history_mask")
    frozen=prior["pooled_fold_queues"]
    if (pooled["positive_cell_days"]!=7209 or
        pooled["novel_positive_cell_days"]!=1664 or
        frozen["simple_neighbor"]["novel_capture"]!=.11946 or
        frozen["synthetic_center_block"]["novel_capture"]!=.13529):
        raise ValueError("First-in-year matched queue changed")
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed_offset":args.seed_offset,
            "question":"Can group-level loss improve the historical-repeat target-mask CNN on the same first-in-year queue?",
            "target":"Traffy report-positive zero-own-365d 250m cell-days; not verified flooding",
            "training":"2023-2024 source districts; actual first-in-year groups plus lower-weight masked repeat groups",
            "target_mask":"Zero four raw report-history channels at every scored target cell for positive members and comparisons",
            "augmentation":"10% random history-pixel dropout and 50% random 3x3 report-history blocks",
            "positive_bag_score":f"{BAG_TEMPERATURE}*logmeanexp(member scores/{BAG_TEMPERATURE})",
            "queue":"10% of heldout fold cells per day; 20% of slots reserved for zero-own-history candidates",
            "comparison":{"nearby_report_capture":frozen[
                "simple_neighbor"]["novel_capture"],
                          "previous_synthetic_center_block_capture":frozen[
                "synthetic_center_block"]["novel_capture"],
                          "previous_source_selected_capture":frozen[
                "source_selected_mask_arm"]["novel_capture"]},
            "folds":records,"pooled_heldout_q4":pooled,
            "limitations":["Unreported candidates are unknown, not verified dry",
                           "Touching cell groups are not original complaint IDs",
                           "Q4 2025 has been inspected repeatedly",
                           "No verified water extent or rain-scenario validity"]}
    out_json.write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"states":states},out_model)
    print("pooled",pooled["top_10pct_capture"],
          pooled["novel_capture"],flush=True)


if __name__=="__main__":
    main()
