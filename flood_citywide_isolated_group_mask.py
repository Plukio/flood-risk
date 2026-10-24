"""Train a spatial ranker with group-level loss and a true target-history mask.

The mask removes the scored cell's 3x3 prior reports from raw history AND from
the five precomputed Gaussian history layers. Positive and unreported
comparison patches receive the same mask. One arm uses actual isolated groups;
the second adds lower-weight, masked first-in-year adjacent-history groups.
Only source-district labels affect training and model-strength selection.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label, maximum_filter
from torch.nn import functional as F

from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_isolated_site_ranker import isolated
from flood_citywide_isolated_context_residual import (
    ALPHAS, ContextResidualCNN, context, pool, score_days,
)
from flood_citywide_spatial_novel_queue import static_image
from flood_citywide_spatial_novel_queue import SpatialNovelCNN
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_isolated_group_target_mask_fivefold.json"
OUT_MODEL = OUT / "q4_2025_isolated_group_target_mask_fivefold.pt"
ARMS = ("isolated_groups", "isolated_plus_masked_adjacent",
        "isolated_plus_masked_repeat_block")
EPOCHS = 2
POS_LIMIT = 12
AUG_LIMIT = 6
BAG_TEMPERATURE = .05
AUG_WEIGHT = .25
REPEAT_WEIGHT = .5
WIDE_LAYERS = ((15, 0, 3), (16, 0, 6), (17, 0, 12),
               (18, 1, 3), (19, 1, 6))


def gaussian_offsets(sigma: int) -> torch.Tensor:
    radius = int(4*sigma + .5)
    offsets = np.arange(-radius, radius+1, dtype=np.float32)
    kernel = np.exp(-.5*(offsets/sigma)**2)
    kernel /= kernel.sum()
    patch_offsets = np.arange(-RADIUS, RADIUS+1)
    values = np.zeros((3, 2*RADIUS+1), np.float32)
    for source, local in enumerate((-1,0,1)):
        delta = patch_offsets-local
        ok = np.abs(delta)<=radius
        values[source,ok] = kernel[delta[ok]+radius]
    return torch.from_numpy(values)


GAUSSIAN = {sigma: gaussian_offsets(sigma) for sigma in (3,6,12)}


def mask_target_history(batch: torch.Tensor) -> tuple[torch.Tensor,dict]:
    """Exact leave-3x3-out correction for Gaussian history in each patch."""
    masked = batch.clone()
    center = batch[:, :4, RADIUS-1:RADIUS+2,
                         RADIUS-1:RADIUS+2]
    audit = {"patches":len(batch),
             "patches_with_local_365d_history":int((center[:,0].sum((1,2))>0).sum()),
             "local_365d_log_count_sum":float(center[:,0].sum())}
    for layer, history, sigma in WIDE_LAYERS:
        weights=GAUSSIAN[sigma].to(batch.device)
        correction=torch.zeros_like(batch[:,layer])
        for i in range(3):
            for j in range(3):
                amount=torch.expm1(center[:,history,i,j])
                correction += (amount[:,None,None] *
                               weights[i][None,:,None] *
                               weights[j][None,None,:])
        raw=torch.expm1(batch[:,layer])
        masked[:,layer]=torch.log1p(torch.clamp(raw-correction,min=0))
    masked[:,:4,RADIUS-1:RADIUS+2,RADIUS-1:RADIUS+2]=0
    if torch.any(masked[:,:4,RADIUS-1:RADIUS+2,
                               RADIUS-1:RADIUS+2]!=0):
        raise ValueError("Target history was not masked")
    return masked,audit


def _shifted_gaussian(sigma: int, source_index: torch.Tensor) -> torch.Tensor:
    radius=int(4*sigma+.5)
    offsets=torch.arange(-radius,radius+1,dtype=torch.float32,
                         device=source_index.device)
    kernel=torch.exp(-.5*(offsets/sigma)**2)
    kernel=kernel/kernel.sum()
    delta=torch.arange(2*RADIUS+1,device=source_index.device)[None,:]-\
          source_index[:,None]
    inside=delta.abs()<=radius
    return torch.where(inside,kernel[(delta+radius).clamp(0,2*radius)],0.)


def mask_random_history_blocks(batch:torch.Tensor,probability:float=.5):
    """Random 3x3 augmentation, including exact smoothed-history removal."""
    chosen=torch.rand(len(batch))<probability
    indices=torch.nonzero(chosen).flatten()
    if not len(indices):
        return batch,0
    rows=torch.randint(0,2*RADIUS-1,(len(indices),))
    cols=torch.randint(0,2*RADIUS-1,(len(indices),))
    selected=batch[indices].clone()
    numbered=torch.arange(len(indices))
    for layer,history,sigma in WIDE_LAYERS:
        correction=torch.zeros_like(selected[:,layer])
        for i in range(3):
            wx=_shifted_gaussian(sigma,rows+i)
            for j in range(3):
                wy=_shifted_gaussian(sigma,cols+j)
                amount=torch.expm1(selected[numbered,history,rows+i,cols+j])
                correction += amount[:,None,None]*wx[:,:,None]*wy[:,None,:]
        selected[:,layer]=torch.log1p(torch.clamp(
            torch.expm1(selected[:,layer])-correction,min=0))
    for k,(row,col) in enumerate(zip(rows.tolist(),cols.tolist())):
        selected[k,:4,row:row+3,col:col+3]=0
    batch[indices]=selected
    return batch,len(indices)


def groups_in_mask(mask: np.ndarray, rng: np.random.Generator,
                   flat_to_index: np.ndarray, limit: int):
    components,count=label(mask,structure=np.ones((3,3),bool))
    ids=np.arange(1,count+1)
    if count>limit:
        ids=rng.choice(ids,size=limit,replace=False)
    flattened=components.ravel()
    return [flat_to_index[np.flatnonzero(flattened==j)] for j in ids]


def sample_group_batch(data:dict,day:int,source_grid:np.ndarray,
                       flat_to_index:np.ndarray,simple:np.ndarray,
                       base_rng:np.random.Generator,
                       aug_rng:np.random.Generator,arm:str):
    truth=data["reports"][day]>0
    known=truth & source_grid
    eligible=isolated(data,day)
    iso_groups=groups_in_mask(known & eligible,base_rng,flat_to_index,POS_LIMIT)
    if not iso_groups:
        return None
    prior=(data["prefix"][day]-data["prefix"][day-365])>0
    no_own=~prior
    aug_groups=[]
    repeat_groups=[]
    if arm=="isolated_plus_masked_adjacent":
        aug_groups=groups_in_mask(
            known & no_own & ~eligible,aug_rng,flat_to_index,AUG_LIMIT)
    if arm=="isolated_plus_masked_repeat_block":
        repeat_groups=groups_in_mask(
            known & prior,aug_rng,flat_to_index,AUG_LIMIT)
    # Never consult target-district same-day outcomes for comparison selection.
    safe=source_grid & ~maximum_filter(known.astype(np.uint8),
                                       size=3,mode="constant").astype(bool)
    scopes=((eligible,iso_groups,base_rng,1.,"isolated"),
            (no_own & ~eligible,aug_groups,aug_rng,AUG_WEIGHT,"adjacent"),
            (eligible,repeat_groups,aug_rng,REPEAT_WEIGHT,"repeat"))
    groups=[]
    for kind_mask,positive_groups,rng,weight,kind in scopes:
        if not positive_groups:
            continue
        pool=np.flatnonzero((safe & kind_mask).ravel())
        if not len(pool):
            continue
        candidates=flat_to_index[pool]
        hard_n=min(len(pool),max(50,len(pool)//5))
        hard=candidates[np.argpartition(simple[candidates],-hard_n)[-hard_n:]]
        for members in positive_groups:
            random=int(rng.choice(candidates))
            difficult=int(rng.choice(hard))
            groups.append((members,np.array([random,difficult],np.int32),
                           weight,kind))
    return groups or None


def transfer_path_spatial_weights(model:ContextResidualCNN,
                                  pretrained:SpatialNovelCNN):
    """Map the 16-channel path trunk into the 21-channel masked ranker."""
    with torch.no_grad():
        model.entry.weight.zero_()
        # Four history, ten static, and candidate channels retain their slots.
        model.entry.weight[:,:15].copy_(pretrained.entry.weight[:,:15])
        # The synthetic source occupied the 365-day history channel only.
        model.entry.weight[:,1:4].zero_()
        # Five wider Gaussian-history channels start at zero. The grid-valid
        # channel moves from index 15 to index 20 in ContextResidualCNN.
        model.entry.weight[:,20].copy_(pretrained.entry.weight[:,15])
        model.entry.bias.copy_(pretrained.entry.bias)
        model.steps.load_state_dict(pretrained.steps.state_dict())
        # ContextResidualCNN initializes its report residual head to zero.
        # Keep that initialization rather than transferring synthetic scores.


def train(data:dict,source:np.ndarray,static:np.ndarray,seed:int,arm:str,
          pretrained:SpatialNovelCNN|None=None,
          model_factory=ContextResidualCNN,auxiliary_step=None,
          context_fn=context):
    torch.manual_seed(seed)
    base_rng=np.random.default_rng(seed)
    aug_rng=np.random.default_rng(seed+913_721)
    source_grid=np.zeros(SHAPE,bool)
    source_grid.ravel()[data["candidate_flat"]]=source
    flat_to_index=np.full(np.prod(SHAPE),-1,np.int32)
    flat_to_index[data["candidate_flat"]]=np.arange(len(data["candidate_flat"]))
    days=[d for d in range(TRAIN_START,VALID_START)
          if np.any((data["reports"][d]>0)&isolated(data,d)&source_grid)]
    model=model_factory()
    if pretrained is not None:
        transfer_path_spatial_weights(model,pretrained)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.02)
    epoch_loss=[]
    stats={"isolated_groups":0,"adjacent_groups":0,"repeat_groups":0,
           "masked_patches_with_local_365d_history":0,
           "random_3x3_blocked_patches":0}
    batch_number=0
    for epoch in range(EPOCHS):
        model.train()
        losses=[]
        for day in base_rng.permutation(days):
            image,_,simple=context_fn(data,int(day),static)
            groups=sample_group_batch(data,int(day),source_grid,
                                      flat_to_index,simple,base_rng,aug_rng,arm)
            if not groups:
                continue
            positive=np.concatenate([g[0] for g in groups])
            comparison=np.concatenate([g[1] for g in groups])
            batch=patches(image,data["candidate_flat"],
                          np.r_[positive,comparison])
            batch,audit=mask_target_history(batch)
            stats["masked_patches_with_local_365d_history"] += audit[
                "patches_with_local_365d_history"]
            if arm=="isolated_plus_masked_repeat_block":
                batch,blocked=mask_random_history_blocks(batch)
                stats["random_3x3_blocked_patches"]+=blocked
            residual=model.patch_scores(batch)
            # The fixed score must also exclude the masked local reports.
            fixed=(.6*batch[:,15,RADIUS,RADIUS]+
                   .3*batch[:,16,RADIUS,RADIUS]+
                   .1*batch[:,17,RADIUS,RADIUS])
            score=fixed+residual
            bag_losses=[]
            start=0
            for j,(members,_,weight,kind) in enumerate(groups):
                n=len(members)
                site=(torch.logsumexp(score[start:start+n]/BAG_TEMPERATURE,
                                      dim=0)-math.log(n))*BAG_TEMPERATURE
                start+=n
                negative=score[len(positive)+2*j:len(positive)+2*j+2]
                bag_losses.append(weight*F.softplus(negative-site).mean())
                stats[kind+"_groups"]+=1
            weights=sum(g[2] for g in groups)
            loss=torch.stack(bag_losses).sum()/weights
            loss=loss+.001*residual.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_number+=1
            if auxiliary_step is not None:
                auxiliary_step(model,optimizer,batch_number)
            losses.append(float(loss.detach()))
        epoch_loss.append(round(float(np.mean(losses)),5))
        print(arm,"epoch",epoch+1,"loss",epoch_loss[-1],flush=True)
    model.eval()
    stats["epoch_group_pairwise_loss"]=epoch_loss
    stats["train_days_with_isolated_positive"]=len(days)
    if arm=="isolated_plus_masked_adjacent" and (
        stats["adjacent_groups"]==0 or
        stats["masked_patches_with_local_365d_history"]==0):
        raise ValueError("Augmented target masks did not remove any history")
    if arm=="isolated_plus_masked_repeat_block" and (
        stats["repeat_groups"]==0 or
        stats["random_3x3_blocked_patches"]==0 or
        stats["masked_patches_with_local_365d_history"]==0):
        raise ValueError("Augmented target masks did not remove any history")
    return model,stats


def mask_self_check(data:dict,static:np.ndarray):
    day=TEST_START
    image,_,_=context(data,day,static)
    prior365=(data["prefix"][day]-data["prefix"][day-365]).astype(np.float32)
    candidate=data["candidate_flat"]
    nonisolated=np.flatnonzero(prior365.ravel()[candidate]>0)
    chosen=nonisolated[len(nonisolated)//2]
    rr,cc=np.unravel_index(candidate[chosen],SHAPE)
    batch=patches(image,candidate,np.array([chosen]))
    masked,audit=mask_target_history(batch)
    if audit["patches_with_local_365d_history"]!=1:
        raise ValueError("Synthetic mask check chose an empty target")
    altered=prior365.copy()
    altered[max(0,rr-1):rr+2,max(0,cc-1):cc+2]=0
    expected=np.log1p(gaussian_filter(altered,3,mode="constant"))[rr,cc]
    if not np.isclose(masked[0,15,RADIUS,RADIUS],expected,atol=2e-6):
        raise ValueError("Gaussian target exclusion differs from recomputation")
    # Isolated targets need no per-target inference pass: their masked inputs
    # already equal their unmasked pre-day inputs.
    eligible=np.flatnonzero(isolated(data,day).ravel()[candidate])
    check=patches(image,candidate,eligible[:32])
    changed,_=mask_target_history(check)
    if not torch.allclose(changed,check,atol=2e-6):
        raise ValueError("Isolated full-grid inference differs from target mask")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed-offset",type=int,default=0)
    parser.add_argument("--arm",choices=("all",*ARMS),default="all")
    args=parser.parse_args()
    active_arms=ARMS if args.arm=="all" else (args.arm,)
    suffix="" if args.seed_offset==0 and args.arm=="all" else (
        f"_seed_{args.seed_offset}_{args.arm}")
    out_json=OUT_JSON if not suffix else OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    out_model=OUT_MODEL if not suffix else OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    data=load_grid()
    district=data["district_id"].ravel()[data["candidate_flat"]]
    folds=np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])),5)
    initial_static,_=static_image(data,np.ones(len(district),bool))
    mask_self_check(data,initial_static)
    records,states=[],{}
    for fold_id,ids in enumerate(folds,1):
        source=~np.isin(district,ids)
        static,norm=static_image(data,source)
        print("fold",fold_id,flush=True)
        record={"fold":fold_id,
                "heldout_district_codes":[data["districts"][j]["code"]
                                          for j in ids],
                "normalization":norm,"arms":{}}
        states[fold_id]={}
        for arm in active_arms:
            model,fit=train(data,source,static,
                            SEED+fold_id*1000+args.seed_offset,arm)
            valid=score_days(data,source,static,model,
                             range(VALID_START,TEST_START))
            chosen=max(ALPHAS,key=lambda a:(
                valid[str(a)]["isolated_group_capture"],
                valid[str(a)]["isolated_cell_capture"],-a))
            test=score_days(data,~source,static,model,
                            range(TEST_START,N_DAYS))
            record["arms"][arm]={"fit":fit,"source_validation":valid,
                                 "selected_alpha":chosen,"heldout_q4":test}
            states[fold_id][arm]={k:v.cpu() for k,v in model.state_dict().items()}
            print("fold",fold_id,arm,"alpha",chosen,"q4 selected",
                  test[str(chosen)]["isolated_cell_capture"],
                  test[str(chosen)]["isolated_group_capture"],flush=True)
        records.append(record)
    pooled={}
    for arm in active_arms:
        pooled[arm]={str(a):pool([f["arms"][arm]["heldout_q4"][str(a)]
                                  for f in records]) for a in ALPHAS}
        pooled[arm]["source_selected"]=pool([
            f["arms"][arm]["heldout_q4"][str(f["arms"][arm]["selected_alpha"])]
            for f in records])
        baseline=pooled[arm]["0.0"]
        if (baseline["all_positive"]!=7209 or
            baseline["isolated_positive"]!=332 or
            baseline["isolated_groups"]!=158 or
            baseline["isolated_hits"]!=65. or
            baseline["group_hits"]!=44.):
            raise ValueError("Frozen queue comparator changed")
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed_offset":args.seed_offset,
            "arms":active_arms,
            "question":"Does connected-group loss plus a true target-history mask improve isolated-site ranking?",
            "target":"Traffy report-positive 250m cell-days, not verified flooding",
            "mask":"Remove scored 3x3 local reports from four raw history channels and all five Gaussian-smoothed history channels, symmetrically for positives and unreported comparisons",
            "positive_bag_score":f"{BAG_TEMPERATURE}*logmeanexp(member scores/{BAG_TEMPERATURE})",
            "augmented_group_weight":AUG_WEIGHT,
            "repeat_group_weight":REPEAT_WEIGHT,
            "queue":"10% of heldout fold cells per day, 20% of slots reserved for isolated candidates",
            "dates":{"training":["2023-01-01","2024-12-31"],
                     "source_validation":["2025-01-01","2025-09-30"],
                     "heldout_q4":["2025-10-01","2025-12-31"]},
            "alpha_candidates":ALPHAS,
            "alpha_selection":"source validation isolated group capture, then cell capture, then smaller alpha",
            "folds":records,"pooled_heldout_q4":pooled,
            "limitations":["Unreported comparison cells are unknown, not verified dry",
                           "Touching positive groups are not original complaint IDs",
                           "Q4 2025 has been inspected repeatedly",
                           "No verified flood extent or rainfall counterfactual"]}
    out_json.write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"states":states},out_model)
    print("pooled",{arm:{k:(v["isolated_hits"],v["group_hits"])
                          for k,v in rows.items()}
                    for arm,rows in pooled.items()},flush=True)


if __name__=="__main__":
    main()
