"""Test whether synthetic center masks leak nearby repeat-site history.

Matched arms hide the same number of training history pixels. One always hides
the target's 3x3 neighborhood; the control hides a random 3x3 block. Both use
the same training pairs, initialization, random 10% history dropout, source
validation dates and held-out district queues.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import label
from torch.nn import functional as F

from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_spatial_novel_queue import (
    MASK_RATE, SpatialNovelCNN, collect, day_maps, queue_weights, static_image,
    top_weights,
)
from flood_citywide_spatial_mask_augmentation import (
    OUT_JSON as PREVIOUS_JSON, OUT_MODEL as PREVIOUS_MODEL,
)
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval

OUT_JSON = OUT / "q4_2025_target_site_mask_fivefold.json"
OUT_MODEL = OUT / "q4_2025_target_site_mask_fivefold.pt"
OUT_CHECKPOINT = OUT / "q4_2025_target_site_mask_partial.pt"
ARMS = ("random_block_matched", "target_centered_3x3")
QUOTA = .20
EPOCHS = 2


def apply_matched_blocks(batch: torch.Tensor, n_positive_pairs: int,
                         n_actual: int, mode: str, rng: np.random.Generator) -> dict:
    """Give each positive/comparison pair the same history-only 3x3 hole.

    Synthetic repeat-site pairs always get the hole; actual first-in-year
    pairs get it half the time. The random control has exactly the same number
    and size of holes, positioned independently of the target center.
    """
    apply = rng.random(n_positive_pairs) < .5
    apply[2*n_actual:] = True
    # Draw positions for both arms so the RNG stream stays matched.
    rr = rng.integers(0,2*RADIUS-1,size=n_positive_pairs)
    cc = rng.integers(0,2*RADIUS-1,size=n_positive_pairs)
    for i in np.flatnonzero(apply):
        row,col = ((RADIUS-1,RADIUS-1) if mode=="target_centered_3x3"
                   else (int(rr[i]),int(cc[i])))
        batch[i,:4,row:row+3,col:col+3] = 0
        batch[n_positive_pairs+i,:4,row:row+3,col:col+3] = 0
    return {"paired_holes":int(apply.sum()),
            "synthetic_pairs_with_hole":int(apply[2*n_actual:].sum())}


def self_check() -> None:
    rng = np.random.default_rng(30)
    original = torch.ones(12,16,15,15)
    center = original.clone()
    random = original.clone()
    a = apply_matched_blocks(center,6,2,"target_centered_3x3",
                             np.random.default_rng(30))
    b = apply_matched_blocks(random,6,2,"random_block_matched",
                             np.random.default_rng(30))
    if a != b or a["synthetic_pairs_with_hole"] != 2:
        raise ValueError("Matched masks changed pair counts")
    if not torch.all(center[4:6,:4,6:9,6:9] == 0):
        raise ValueError("Synthetic-site neighbor history remains visible")
    if not torch.equal(center[:,4:],original[:,4:]):
        raise ValueError("Physical layers were masked")
    if not torch.equal(random[:,4:],original[:,4:]):
        raise ValueError("Random control changed physical layers")


def fit(data: dict, source: np.ndarray, static: np.ndarray, seed: int,
        mode: str):
    torch.manual_seed(seed)
    sample_rng = np.random.default_rng(seed)
    mask_rng = np.random.default_rng(seed+913_721)
    model = SpatialNovelCNN()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.02)
    flat = data["candidate_flat"]
    days = []
    for day in range(TRAIN_START,VALID_START):
        truth = data["reports"][day].ravel()[flat] > 0
        if np.any(truth & source):
            days.append(day)
    losses_by_epoch,holes,synthetic_pairs = [],0,0
    for epoch in range(1,EPOCHS+1):
        model.train()
        losses = []
        for day in sample_rng.permutation(days):
            image,_,simple,novel = day_maps(data,int(day),static)
            truth = data["reports"][day].ravel()[flat] > 0
            actual = np.flatnonzero(novel & truth & source)
            known = np.flatnonzero(~novel & truth & source)
            unknown = np.flatnonzero(novel & ~truth & source)
            if not len(unknown):
                continue
            a = sample_rng.choice(actual,size=min(16,len(actual)),replace=False)
            s = sample_rng.choice(known,size=min(8,len(known)),replace=False)
            pos = np.r_[a,s]
            if not len(pos):
                continue
            hard_count = min(len(unknown),max(50,len(unknown)//5))
            hard_pool = unknown[np.argpartition(simple[unknown],
                                      len(unknown)-hard_count)[-hard_count:]]
            cmp_random = sample_rng.choice(unknown,size=len(pos),replace=True)
            cmp_hard = sample_rng.choice(hard_pool,size=len(pos),replace=True)
            positive_indices = np.repeat(pos,2)
            comparison_indices = np.column_stack((cmp_random,cmp_hard)).ravel()
            batch = patches(image,flat,np.r_[positive_indices,comparison_indices])
            # Only synthetic repeat-site positives had a nonzero center. The
            # original four pre-day counts are hidden before spatial masks.
            batch[2*len(a):2*len(pos),:4,RADIUS,RADIUS] = 0
            if torch.any(batch[:,:4,RADIUS,RADIUS] != 0):
                raise ValueError("Own report history exposed in training")
            keep = (torch.rand(len(batch),1,2*RADIUS+1,
                               2*RADIUS+1) >= MASK_RATE).float()
            batch[:,:4] *= keep
            audit = apply_matched_blocks(batch,2*len(pos),len(a),mode,mask_rng)
            holes += audit["paired_holes"]
            synthetic_pairs += audit["synthetic_pairs_with_hole"]
            scores = model.patch_scores(batch)
            per_pair = F.softplus(scores[2*len(pos):]-scores[:2*len(pos)])
            weights = torch.cat((torch.ones(2*len(a)),
                                 torch.full((2*len(s),),.5)))
            loss = (per_pair*weights).sum()/weights.sum()
            if not len(a):
                loss = .5*loss
            loss += .0005*scores.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        losses_by_epoch.append(round(float(np.mean(losses)),5))
        print(mode,"epoch",epoch,"loss",losses_by_epoch[-1],flush=True)
    model.eval()
    return model,{"training_days":len(days),"epoch_pairwise_loss":losses_by_epoch,
                  "paired_holes":holes,"synthetic_positive_pairs_masked":synthetic_pairs,
                  "mask_policy":"all synthetic pairs and half actual-novel pairs",
                  "random_history_pixel_mask_rate":MASK_RATE}


def q20_rows(data: dict, arrays, scope: np.ndarray, clusters: bool):
    base,simple,learned,novel,truth = arrays
    indices = np.flatnonzero(scope)
    full_flat = data["candidate_flat"][indices]
    rows = {name:{"daily_positive":[],"daily_captured":[],
                  "daily_novel_positive":[],"daily_novel_captured":[],
                  "daily_clusters":[],"daily_cluster_hits":[]}
            for name in ("fixed_full","simple_neighbor","model")}
    for day in range(len(truth)):
        b,n,y = base[day,indices],novel[day,indices],truth[day,indices]
        picks = {"fixed_full":top_weights(b,.1*len(indices)),
                 "simple_neighbor":queue_weights(b,simple[day,indices],n,QUOTA),
                 "model":queue_weights(b,learned[day,indices],n,QUOTA)}
        if clusters:
            grid = np.zeros(SHAPE,bool)
            grid.ravel()[full_flat] = y > 0
            components,count = label(grid,structure=np.ones((3,3),bool))
            ids = components.ravel()[full_flat]
            members = [ids==j for j in range(1,count+1)]
        for name,w in picks.items():
            row=rows[name]
            row["daily_positive"].append(int(y.sum()))
            row["daily_captured"].append(round(float(np.dot(w,y)),5))
            row["daily_novel_positive"].append(int(y[n].sum()))
            row["daily_novel_captured"].append(round(float(np.dot(w,n*y)),5))
            if clusters:
                row["daily_clusters"].append(count)
                row["daily_cluster_hits"].append(round(
                    sum(float(w[m].max()) for m in members),5))
    out={}
    for name,row in rows.items():
        positive=sum(row["daily_positive"])
        novel_positive=sum(row["daily_novel_positive"])
        cluster_count=sum(row["daily_clusters"])
        out[name]={"candidate_cells":int(scope.sum()),
                   "positive_cell_days":positive,
                   "top_10pct_capture":round(sum(row["daily_captured"])/positive,5),
                   "novel_positive_cell_days":novel_positive,
                   "novel_capture":round(sum(row["daily_novel_captured"])/novel_positive,5),
                   "same_day_adjacent_clusters":cluster_count if clusters else None,
                   "cluster_capture":round(sum(row["daily_cluster_hits"])/cluster_count,5)
                       if cluster_count else None,
                   **{k:v for k,v in row.items() if k.startswith("daily_")}}
    return out


def aggregate(folds: list[dict],name: str):
    rows=[f["q4"][name] for f in folds]
    pos=sum(r["positive_cell_days"] for r in rows)
    novel=sum(r["novel_positive_cell_days"] for r in rows)
    clusters=sum(r["same_day_adjacent_clusters"] for r in rows)
    hits=sum(sum(r["daily_captured"]) for r in rows)
    novel_hits=sum(sum(r["daily_novel_captured"]) for r in rows)
    # The frozen previous benchmark stores only its aggregate cluster capture.
    cluster_hits=sum(sum(r["daily_cluster_hits"]) if "daily_cluster_hits" in r
                     else r["cluster_capture"]*r["same_day_adjacent_clusters"]
                     for r in rows)
    daily_pos=np.sum([r["daily_positive"] for r in rows],axis=0)
    daily_hits=np.sum([r["daily_captured"] for r in rows],axis=0)
    daily_novel_pos=np.sum([r["daily_novel_positive"] for r in rows],axis=0)
    daily_novel_hits=np.sum([r["daily_novel_captured"] for r in rows],axis=0)
    return {"candidate_cells":sum(r["candidate_cells"] for r in rows),
            "positive_cell_days":pos,"top_10pct_capture":round(hits/pos,5),
            "novel_positive_cell_days":novel,"novel_capture":round(novel_hits/novel,5),
            "same_day_adjacent_clusters":clusters,
            "cluster_capture":round(cluster_hits/clusters,5),
            "daily_positive":daily_pos.astype(int).tolist(),
            "daily_captured":np.round(daily_hits,5).tolist(),
            "daily_novel_positive":daily_novel_pos.astype(int).tolist(),
            "daily_novel_captured":np.round(daily_novel_hits,5).tolist()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    self_check()
    data=load_grid()
    previous=json.loads(PREVIOUS_JSON.read_text())
    old_states=torch.load(PREVIOUS_MODEL,weights_only=True)["states"]
    district=data["district_id"].ravel()[data["candidate_flat"]]
    district_folds=np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])),5)
    if OUT_CHECKPOINT.exists():
        checkpoint=torch.load(OUT_CHECKPOINT,weights_only=False)
        folds,states=checkpoint["folds"],checkpoint["states"]
        print("resuming after",len(folds),"completed folds",flush=True)
    else:
        folds,states=[],{}
    for fold_number,ids in enumerate(district_folds,1):
        if fold_number<=len(folds):
            continue
        source=~np.isin(district,ids)
        target=~source
        static,normalization=static_image(data,source)
        models,fits={},{}
        states[fold_number]={}
        for arm in ARMS:
            print("fold",fold_number,arm,flush=True)
            models[arm],fits[arm]=fit(data,source,static,
                                      SEED+fold_number*1000,arm)
            states[fold_number][arm]={k:v.cpu() for k,v in
                                      models[arm].state_dict().items()}
        validation={}
        for arm,model in models.items():
            arrays=collect(data,model,static,range(VALID_START,TEST_START))
            row=q20_rows(data,arrays,source,False)["model"]
            validation[arm]={"overall_capture":row["top_10pct_capture"],
                             "novel_capture":row["novel_capture"]}
        chosen=max(ARMS,key=lambda arm:validation[arm]["novel_capture"])
        q4={}
        for arm,model in models.items():
            arrays=collect(data,model,static,range(TEST_START,N_DAYS))
            rows=q20_rows(data,arrays,target,True)
            q4[arm]=rows["model"]
            if "simple_neighbor" not in q4:
                q4["simple_neighbor"]=rows["simple_neighbor"]
                q4["fixed_full"]=rows["fixed_full"]
        # Frozen previous arms provide an honest matched comparison.
        for name in ("synthetic_center","synthetic_center_block"):
            prior=previous["folds"][fold_number-1]["q4"][name]
            q4[name]=prior
        if (q4["simple_neighbor"]["top_10pct_capture"] !=
            previous["folds"][fold_number-1]["q4"]["simple_neighbor"]["top_10pct_capture"]):
            raise ValueError("Simple neighbor queue changed")
        folds.append({"fold":fold_number,
                      "district_codes":[data["districts"][j]["code"] for j in ids],
                      "normalization":normalization,"fit":fits,
                      "source_validation":validation,"source_selected_arm":chosen,
                      "q4":q4})
        print("fold",fold_number,"source chose",chosen,
              {name:(row["top_10pct_capture"],row["novel_capture"])
               for name,row in q4.items()},flush=True)
        torch.save({"folds":folds,"states":states},OUT_CHECKPOINT)
    names=("fixed_full","simple_neighbor","synthetic_center",
           "synthetic_center_block",*ARMS)
    pooled={name:aggregate(folds,name) for name in names}
    if (pooled["fixed_full"]["positive_cell_days"]!=7209 or
        pooled["fixed_full"]["novel_positive_cell_days"]!=1664 or
        pooled["fixed_full"]["top_10pct_capture"]!=.47113 or
        pooled["simple_neighbor"]["top_10pct_capture"]!=.44114 or
        pooled["synthetic_center_block"]["top_10pct_capture"]!=.44479):
        raise ValueError("Frozen Q4 comparison changed")
    selected_rows=[f["q4"][f["source_selected_arm"]] for f in folds]
    positive=sum(r["positive_cell_days"] for r in selected_rows)
    novel=sum(r["novel_positive_cell_days"] for r in selected_rows)
    clusters=sum(r["same_day_adjacent_clusters"] for r in selected_rows)
    daily_positive=np.sum([r["daily_positive"] for r in selected_rows],axis=0)
    daily_hits=np.sum([r["daily_captured"] for r in selected_rows],axis=0)
    daily_novel=np.sum([r["daily_novel_positive"] for r in selected_rows],axis=0)
    daily_novel_hits=np.sum([r["daily_novel_captured"] for r in selected_rows],axis=0)
    pooled["source_selected_location"]={
        "positive_cell_days":positive,
        "top_10pct_capture":round(float(daily_hits.sum()/positive),5),
        "novel_positive_cell_days":novel,
        "novel_capture":round(float(daily_novel_hits.sum()/novel),5),
        "same_day_adjacent_clusters":clusters,
        "cluster_capture":round(sum(sum(r["daily_cluster_hits"]) for r in selected_rows)/clusters,5),
        "daily_positive":daily_positive.astype(int).tolist(),
        "daily_captured":np.round(daily_hits,5).tolist(),
        "daily_novel_positive":daily_novel.astype(int).tolist(),
        "daily_novel_captured":np.round(daily_novel_hits,5).tolist()}
    pooled["selected_minus_random_weekly_interval"]=paired_interval(
        pooled["source_selected_location"],pooled["random_block_matched"])
    output={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question":"Does target-centered historical masking remove an adjacent-report shortcut?",
            "target":"first-in-a-year Traffy report-positive 250m cell-days, not verified flood",
            "quota_share_of_total_inspection_slots":QUOTA,
            "training_period":["2023-01-01","2024-12-31"],
            "source_validation_period":["2025-01-01","2025-09-30"],
            "heldout_test_period":["2025-10-01","2025-12-31"],
            "mask_arms":ARMS,"folds":folds,"pooled_fold_queues":pooled,
            "limitations":["Adjacent earlier reports are not linked to original complaint IDs",
                           "Random masks alter synthetic training inputs, not real missingness",
                           "Unreported cells are unknown, not dry",
                           "Q4 has been inspected repeatedly in project work",
                           "Physical layers are static snapshots; no verified hydraulic links"]}
    OUT_JSON.write_text(json.dumps(output,indent=2)+"\n")
    torch.save({"states":states},OUT_MODEL)
    OUT_CHECKPOINT.unlink(missing_ok=True)
    print("pooled",{n:(pooled[n]["top_10pct_capture"],pooled[n]["novel_capture"])
                    for n in names+("source_selected_location",)},flush=True)
    print("saved",OUT_JSON,OUT_MODEL,flush=True)


if __name__=="__main__":
    main()
