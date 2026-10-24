"""Matched target-centered versus random history masks in a joint CNN.

The satellite scene heads train on external UNOSAT, EOS and GISTDA weak labels.
The Traffy head trains on positive-vs-unreported relative pairs, with own-site
report history hidden in both arms. Only the extra 3x3 block location differs.
Pathum Thani satellite scenes and Bangkok EOS scenes are excluded from fitting.
"""
from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F

from flood_citywide_history_ranker import VALID_START
from flood_label_audit import scene_path, validate_scene
from flood_multi_target_data import PATCH_CODES
from flood_multi_target_train import (
    HEAD_FOR, TASK_PROBS, MultiTargetCNN, load_urban, location_loss,
    urban_sample,
)
from flood_open_weak_data import FEATURE_NAMES, WEATHER_NAMES, OUTPUT, ROOT
from flood_open_weak_train import augment, weak_loss

DEST=OUTPUT / "multi_target" / "target_site_mask"
ARMS=("random_block_matched","target_centered_3x3")
CONTROL_ARMS=("urban_only_random_block_matched",
              "urban_only_target_centered_3x3")
HOLDOUT_REGIONS={"pathum_thani","nong_chok","bang_kapi_core"}
SEED=18319
EPOCHS=64
STEPS_PER_EPOCH=48
MASK_RATE=.10


def scenes_and_norm() -> tuple[dict,dict,list[dict],list[dict]]:
    manifest=json.loads((OUTPUT/"data_audit.json").read_text())
    if tuple(manifest["features"])!=FEATURE_NAMES or tuple(manifest["weather"])!=WEATHER_NAMES:
        raise ValueError("Satellite feature order changed")
    training,holdout=[],[]
    for entry in manifest["scenes"]:
        with np.load(scene_path(entry),allow_pickle=False) as z:
            scene={key:z[key].astype(np.float32) for key in
                   ("static","weather","wet","observed")}
        validate_scene(entry,scene,manifest)
        (holdout if entry["region"] in HOLDOUT_REGIONS else training).append(
            {**entry,**scene})
    unique={s["region"]:s["static"] for s in training}
    static=np.stack(list(unique.values()))
    weather=np.stack([s["weather"] for s in training])
    sx_mean=static.mean(axis=(0,2,3)).astype(np.float32)
    sx_std=np.maximum(static.std(axis=(0,2,3)),.02).astype(np.float32)
    wx_mean=weather.mean(axis=0).astype(np.float32)
    wx_std=np.maximum(weather.std(axis=0),.05).astype(np.float32)
    norm={"static_mean":sx_mean.tolist(),"static_std":sx_std.tolist(),
          "weather_mean":wx_mean.tolist(),"weather_std":wx_std.tolist(),
          "distinct_regions":len(unique)}
    raw_holdout=holdout
    for s in training:
        s["static"]=np.clip((s["static"]-sx_mean[:,None,None])/
                             sx_std[:,None,None],-5,5)
        s["weather"]=np.clip((s["weather"]-wx_mean)/wx_std,-5,5)
    counts=Counter(s["source"] for s in training)
    if set(counts)!=set(HEAD_FOR)-{"eos2026"}:
        raise ValueError(f"Unexpected satellite training sources: {counts}")
    return manifest,norm,training,raw_holdout


def paired_report_loss(net:MultiTargetCNN,spatial:torch.Tensor,
                       weather:torch.Tensor,history:torch.Tensor,
                       reports:torch.Tensor,mask:torch.Tensor,
                       sample_rng:np.random.Generator,
                       mask_rng:np.random.Generator,mode:str) -> tuple[torch.Tensor,str]:
    prior=history[0,0].numpy()
    positive=reports.numpy()>0
    valid=mask.numpy().astype(bool)
    actual=np.flatnonzero((positive & valid & (prior==0)).ravel())
    synthetic=np.flatnonzero((positive & valid & (prior>0)).ravel())
    unknown=np.flatnonzero((~positive & valid & (prior==0)).ravel())
    if not len(unknown) or not (len(actual) or len(synthetic)):
        raise ValueError("Urban pair lacks positive or unreported comparison")
    use_actual=bool(len(actual) and (not len(synthetic) or sample_rng.random()<.5))
    pos=int(sample_rng.choice(actual if use_actual else synthetic))
    if sample_rng.random()<.5:
        nearby=gaussian_filter(prior,1,mode="constant").ravel()[unknown]
        cutoff=max(1,len(unknown)//5)
        pool=unknown[np.argpartition(nearby,len(unknown)-cutoff)[-cutoff:]]
        cmp=int(sample_rng.choice(pool))
    else:
        cmp=int(sample_rng.choice(unknown))
    pr,pc=divmod(pos,80)
    cr,cc=divmod(cmp,80)
    h=history.repeat(2,1,1,1).clone()
    dropout=(torch.rand((1,1,80,80))>=MASK_RATE).float()
    h[:,:1]*=dropout
    h[0,0,pr,pc]=0
    h[1,0,cr,cc]=0
    # Draw both random locations in both arms to preserve the mask RNG stream.
    random_rows=mask_rng.integers(0,78,size=2)
    random_cols=mask_rng.integers(0,78,size=2)
    for i,(row,col) in enumerate(((pr,pc),(cr,cc))):
        if mode=="target_centered_3x3":
            r=max(0,min(77,row-1))
            c=max(0,min(77,col-1))
        else:
            r=int(random_rows[i]);c=int(random_cols[i])
        h[i,0,r:r+3,c:c+3]=0
    prediction=net(spatial.repeat(2,1,1,1),
                   weather.repeat(2,1),h)["traffy"]
    a,b=prediction[0,pr,pc],prediction[1,cr,cc]
    loss=F.softplus(b-a)+.0005*(a.square()+b.square())/2
    return loss,"actual_first_in_year" if use_actual else "synthetic_repeat"


def self_check() -> None:
    # With just one nonzero neighboring pixel, only the centered block removes it.
    h=torch.zeros((2,2,80,80))
    h[:,0,40,41]=1
    h[0,0,40,40]=0
    h[1,0,40,40]=0
    centered=h.clone()
    centered[:,0,39:42,39:42]=0
    if centered[:,0,40,41].sum()!=0 or h[:,0,40,41].sum()!=2:
        raise ValueError("Centered block fails to hide adjacent report")


def train_one(mode:str,manifest:dict,norm:dict,satellite:list[dict],
              urban:dict,satellite_enabled:bool=True) -> dict:
    mask_mode=mode.removeprefix("urban_only_")
    if mask_mode not in ARMS:
        raise ValueError(f"Unknown mask mode: {mode}")
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    mask_rng=np.random.default_rng(SEED+913721)
    torch.set_num_threads(4)
    groups=defaultdict(list)
    for scene in satellite:
        groups[scene["source"]].append(scene)
    task_probs={key:value for key,value in TASK_PROBS.items()
                if key in groups or key in ("traffy","bma")}
    total=sum(task_probs.values())
    tasks=tuple(task_probs)
    probs=np.asarray([task_probs[key]/total for key in tasks])
    report_regions=list(urban)
    bma_regions=[key for key,v in urban.items()
                 if np.any(v["bma_days"]<VALID_START)]
    if not bma_regions or set(report_regions)!=set(PATCH_CODES):
        raise ValueError("Urban training regions changed")
    model=MultiTargetCNN()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=EPOCHS,eta_min=.0001)
    counts=Counter()
    curve=[]
    for epoch in range(1,EPOCHS+1):
        model.train()
        losses=defaultdict(list)
        for _ in range(STEPS_PER_EPOCH):
            task=str(rng.choice(tasks,p=probs))
            optimizer.zero_grad()
            if task in HEAD_FOR:
                pool=groups[task]
                if task=="unosat2021":
                    dry=[s for s in pool if s["wet"].sum()<=1]
                    wet=[s for s in pool if s["wet"].sum()>1]
                    pool=dry if dry and rng.random()<.15 else wet
                scene=pool[int(rng.integers(len(pool)))]
                spatial,weather,wet,observed=augment(scene,rng)
                if not satellite_enabled:
                    counts[f"skipped_{task}"]+=1
                    continue
                empty=torch.zeros((1,2,80,80),dtype=torch.float32)
                prediction=model(spatial,weather,empty)[HEAD_FOR[task]][0]
                loss,_=weak_loss(prediction,wet,observed,task)
            else:
                region=str(rng.choice(bma_regions if task=="bma" else report_regions))
                item=urban[region]
                possible=item["bma_days"] if task=="bma" else item["report_days"]
                days=possible[possible<VALID_START]
                day=int(rng.choice(days))
                spatial,weather,history,reports,depths,mask=urban_sample(item,day,rng)
                if task=="traffy":
                    loss,kind=paired_report_loss(model,spatial,weather,history,
                         reports,mask,rng,mask_rng,mask_mode)
                    counts[kind]+=1
                else:
                    prediction=model(spatial,weather,history)
                    positive=torch.isfinite(depths).float()
                    rank=location_loss(prediction["bma"][0],positive,mask)
                    valid=torch.isfinite(depths)
                    severity=F.smooth_l1_loss(
                        (prediction["depth_cm"][0][valid]-depths[valid])/10,
                        torch.zeros_like(depths[valid]),beta=.5)
                    loss=rank+.2*severity
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {task} loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            counts[task]+=1
            losses[task].append(float(loss.detach()))
        curve.append({"epoch":epoch,
                      **{k:round(float(np.mean(v)),5) for k,v in losses.items()}})
        scheduler.step()
        if epoch==1 or epoch%8==0 or epoch==EPOCHS:
            print(mode,"epoch",epoch,curve[-1],flush=True)
    checkpoint=DEST/f"{mode}.pt"
    torch.save({"state_dict":model.state_dict(),"model":"MultiTargetCNN",
                "normalization":norm,"mask_mode":mask_mode,"seed":SEED,
                "satellite_training_enabled":satellite_enabled,
                "excluded_satellite_regions":sorted(HOLDOUT_REGIONS),
                "urban_train_end":"2024-12-31",
                "target":"source-specific satellite positives and relative urban report/depth sites"},
               checkpoint)
    record={"mode":mode,"seed":SEED,
            "checkpoint":str(checkpoint.relative_to(ROOT)),
            "steps":EPOCHS*STEPS_PER_EPOCH,"task_counts":dict(counts),
            "satellite_training_enabled":satellite_enabled,
            "source_scene_counts":dict(Counter(s["source"] for s in satellite)),
            "epoch_losses":curve}
    (DEST/f"{mode}_train.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


def main(force:bool=False,seed:int=SEED,destination:Path=DEST,
         joint_only:bool=False) -> None:
    global SEED,DEST
    SEED=seed
    DEST=destination.resolve()
    DEST.mkdir(parents=True,exist_ok=True)
    self_check()
    manifest,norm,satellite,holdout=scenes_and_norm()
    urban,urban_counts=load_urban(norm,manifest)
    records=[]
    modes=ARMS if joint_only else (*ARMS,*CONTROL_ARMS)
    for mode in modes:
        checkpoint=DEST/f"{mode}.pt"
        record_path=DEST/f"{mode}_train.json"
        if not force and checkpoint.exists() and record_path.exists():
            saved=torch.load(checkpoint,weights_only=False)
            if saved["seed"]!=SEED or saved.get("satellite_training_enabled",True)!=\
               (not mode.startswith("urban_only_")):
                raise ValueError(f"Saved {mode} differs from requested seed/task mix; use --force")
            record=json.loads(record_path.read_text())
            record["seed"]=SEED
            record_path.write_text(json.dumps(record,indent=2)+"\n")
            records.append(record)
            print("reused",mode,flush=True)
        else:
            records.append(train_one(mode,manifest,norm,satellite,urban,
                       satellite_enabled=not mode.startswith("urban_only_")))
    output={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed":SEED,"matched_arms":ARMS,
            "urban_only_control_arms":() if joint_only else CONTROL_ARMS,
            "holdout_regions":sorted(HOLDOUT_REGIONS),
            "holdout_scene_counts":dict(Counter(s["source"] for s in holdout)),
            "urban_data_counts":urban_counts,"training":records,
            "mask_policy":"Own-site history hidden at positive and unreported comparator; one matched 3x3 block at target or random patch location in each input",
            "unreported_comparison_status":"unknown, not verified dry"}
    (DEST/"training_summary.json").write_text(json.dumps(output,indent=2)+"\n")
    print("saved",DEST,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--force",action="store_true",
                        help="retrain all four matched arms")
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--output-dir",type=Path,default=DEST)
    parser.add_argument("--joint-only",action="store_true")
    args=parser.parse_args()
    main(args.force,args.seed,args.output_dir,args.joint_only)
