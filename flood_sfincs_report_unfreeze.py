"""Unfreeze the final 10 m SFINCS layers with masked 250 m report ranking.

Starting from the directed-neighbor report CNN, only the last two 10 m
convolutions and depth/velocity heads receive report gradients. The report
head stays fixed. Each report patch replaces its central 3x3 pooled physics
cells differentiably, including the target's directed incoming-flow cue.
The first two 5-minute states always come from the original frozen model;
this same convention is used when regenerating 250 m inference features.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import rasterio
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from flood_bangkok_transfer import CACHE as BMA_CACHE
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_isolated_context_residual import ALPHAS, context, score_days
from flood_citywide_isolated_group_mask import (
    BAG_TEMPERATURE, mask_target_history, sample_group_batch,
)
from flood_citywide_isolated_site_ranker import isolated
from flood_citywide_route_pretrain import lidar_static, terrain
from flood_multi_target_citywide_data import OUT
from flood_sfincs_direction_pretrain import FlowMaskCNN
from flood_sfincs_report_direction_route import (
    OUT_JSON as ROUTE_JSON, OUT_MODEL as ROUTE_MODEL, RoutedReportCNN,
    incoming_proxy, normalize_proxy,
)
from flood_sfincs_report_features import (
    CHECKPOINT, OUT_NPZ as FEATURES_NPZ, SPECIAL, build_features,
    cropped_10m, forward_features,
)
from flood_sfincs_report_mask_transfer import (
    apply_physics_transform, make_context, physics_for_fold,
)

OUT_JSON=OUT/"q4_2025_sfincs_report_unfreeze_fivefold.json"
OUT_MODEL=OUT/"q4_2025_sfincs_report_unfreeze_fivefold.pt"
MAX_DAYS=96
GROUPS_PER_DAY=3
MAX_PATCHES=18
ANCHOR_WEIGHT=.005
LEARNING_RATE=.0001


def load_sources(data):
    rasters={}
    for idx,district in enumerate(data["districts"]):
        path=SPECIAL.get(district["code"],BMA_CACHE/f"terrain_{district['code']}.tif")
        if not path.exists():
            continue
        with rasterio.open(path) as source:
            if source.res[0]>15.01:
                continue
        prepared=cropped_10m(path)
        if prepared is not None:
            rasters[idx]=prepared
    return rasters


def crop_index(data,coverage,rasters):
    flat=data["candidate_flat"]
    district=data["district_id"].ravel()[flat]
    rows,cols=np.unravel_index(flat,SHAPE)
    lookup=[]
    eligible=np.zeros(len(flat),bool)
    for j,(r,c) in enumerate(zip(rows,cols)):
        key=int(district[j])
        if not coverage[r,c] or key not in rasters:
            lookup.append(None)
            continue
        z,r0,c0,_=rasters[key]
        sr=(int(r)-r0)*25+12-48
        sc=(int(c)-c0)*25+12-48
        valid=(sr>=0 and sc>=0 and sr+96<=z.shape[0] and
               sc+96<=z.shape[1])
        # Every source cell in the differentiable 3x3 route must be within
        # the same qualifying physics footprint.
        same=(r>0 and c>0 and r<SHAPE[0]-1 and c<SHAPE[1]-1 and
              np.all(coverage[r-1:r+2,c-1:c+2]) and
              np.all(data["district_id"][r-1:r+2,c-1:c+2]==key))
        if valid and same:
            eligible[j]=True
            lookup.append((key,sr,sc))
        else:
            lookup.append(None)
    return lookup,eligible


def static_patch_arrays(indices,lookup,rasters):
    z=np.stack([rasters[lookup[int(j)][0]][0][
        lookup[int(j)][1]:lookup[int(j)][1]+96,
        lookup[int(j)][2]:lookup[int(j)][2]+96] for j in indices])
    centered=np.clip((z-np.median(z,axis=(1,2))[:,None,None])/.5,-6,6)
    relief=np.clip((z-np.stack([gaussian_filter(p,3) for p in z]))/.1,-6,6)
    return centered.astype(np.float32),relief.astype(np.float32)


def final_state(model,reference,centered,relief,device):
    b=len(centered)
    x=torch.zeros((b,7,96,96),dtype=torch.float32,device=device)
    x[:,0]=torch.from_numpy(centered).to(device)
    x[:,1]=torch.from_numpy(relief).to(device)
    x[:,4]=40/60
    previous=torch.zeros_like(x[:,2:3])
    current=torch.zeros_like(previous)
    with torch.no_grad():
        for _ in range(2):
            x[:,2:3]=previous/.05
            x[:,3:4]=current/.05
            following,_,_=forward_features(reference,x)
            previous,current=current,following
    x[:,2:3]=previous/.05
    x[:,3:4]=current/.05
    depth,velocity,hidden=forward_features(model,x)
    return x,depth,velocity,hidden


def pooled_nine(depth,velocity,hidden):
    all_maps=torch.cat((depth*1000,velocity,hidden),1)
    blocks=all_maps[:,:,11:86,11:86].reshape(
        len(all_maps),51,3,25,3,25).permute(0,1,2,4,3,5)
    means=blocks.mean((-1,-2))
    p90=torch.quantile(blocks[:,0].reshape(len(all_maps),3,3,625),.9,dim=-1)
    speed=torch.hypot(blocks[:,1],blocks[:,2]).mean((-1,-2))
    return torch.cat((means[:,0:1],p90[:,None],means[:,1:3],
                      speed[:,None],means[:,3:]),1)  # B,53,3,3


def transform_nine(raw,transform,route_norm,device):
    def constant(name):
        return torch.as_tensor(transform[name],dtype=raw.dtype,device=device)
    summary=((raw[:,:5]-constant("summary_mean")[None,:,None,None])/
             constant("summary_std")[None,:,None,None]).clamp(-5,5)
    hidden=raw[:,5:]-constant("pca_mean")[None,:,None,None]
    components=constant("pca_components")
    projected=torch.einsum("kc,bcrs->bkrs",components,hidden)
    projected=((projected-constant("projected_mean")[None,:,None,None])/
               constant("projected_std")[None,:,None,None]).clamp(-5,5)
    physics=torch.cat((summary,projected),1)
    depth=raw[:,0]*.001
    east=depth*raw[:,2]
    north=depth*raw[:,3]
    incoming=(F.relu(east[:,1,0])+F.relu(-east[:,1,2])+
              F.relu(north[:,2,1])+F.relu(-north[:,0,1]))
    outgoing=east[:,1,1].abs()+north[:,1,1].abs()
    route=torch.stack((incoming,incoming-outgoing),1)
    mean=torch.as_tensor(route_norm["mean"],dtype=raw.dtype,device=device)
    std=torch.as_tensor(route_norm["std"],dtype=raw.dtype,device=device)
    route=((route-mean[None])/std[None]).clamp(-5,5)
    return physics,route


def differentiable_scores(batch,teacher,reference,report_model,indices,
                          lookup,rasters,transform,route_norm,device):
    centered,relief=static_patch_arrays(indices,lookup,rasters)
    x,depth,velocity,hidden=final_state(teacher,reference,centered,relief,device)
    raw=pooled_nine(depth,velocity,hidden)
    physics,route=transform_nine(raw,transform,route_norm,device)
    updated=batch.to(device).clone()
    region=updated[:,20:33,RADIUS-1:RADIUS+2,RADIUS-1:RADIUS+2]
    covered=updated[:,33:34,RADIUS-1:RADIUS+2,RADIUS-1:RADIUS+2]>0
    updated[:,20:33,RADIUS-1:RADIUS+2,RADIUS-1:RADIUS+2]=torch.where(
        covered,physics,region)
    updated[:,34:36,RADIUS,RADIUS]=route
    residual=report_model.patch_scores(updated)
    fixed=(.6*updated[:,15,RADIUS,RADIUS]+.3*updated[:,16,RADIUS,RADIUS]+
           .1*updated[:,17,RADIUS,RADIUS])
    with torch.no_grad():
        old_depth,old_velocity,_=forward_features(reference,x)
    anchor=(((depth-old_depth)/.01).square().mean()+
            .1*((velocity-old_velocity)/.1).square().mean())
    return fixed+residual,anchor


def finetune(data,source,static,physics,route,teacher,reference,report_model,
             lookup,eligible,rasters,transform,route_norm,seed,max_days,device):
    rng=np.random.default_rng(seed)
    source_eligible=source&eligible
    source_grid=np.zeros(SHAPE,bool)
    source_grid.ravel()[data["candidate_flat"]]=source_eligible
    flat_to_index=np.full(np.prod(SHAPE),-1,np.int32)
    flat_to_index[data["candidate_flat"]]=np.arange(len(source))
    days=[d for d in range(TRAIN_START,VALID_START)
          if np.any((data["reports"][d]>0)&isolated(data,d)&source_grid)]
    selected=rng.choice(days,size=min(max_days,len(days)),replace=False)
    for name,parameter in teacher.named_parameters():
        parameter.requires_grad_(name.startswith("layers.8.") or
                                 name.startswith("layers.9.") or
                                 name.startswith("depth_head.") or
                                 name.startswith("velocity_head."))
    report_model.eval()
    for parameter in report_model.parameters():
        parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in teacher.parameters() if p.requires_grad],
                                lr=LEARNING_RATE,weight_decay=.01)
    context_fn=make_context(np.concatenate((physics,route),axis=0))
    losses=[]
    accepted=0
    teacher.train()
    for day in rng.permutation(selected):
        image,_,simple=context_fn(data,int(day),static)
        groups=sample_group_batch(data,int(day),source_grid,flat_to_index,
                                  simple,rng,rng,"isolated_plus_masked_adjacent")
        if not groups:
            continue
        kept=[]
        for g in rng.permutation(len(groups)):
            group=groups[int(g)]
            if sum(len(x[0])+len(x[1]) for x in kept)+len(group[0])+len(group[1])>MAX_PATCHES:
                continue
            kept.append(group)
            if len(kept)>=GROUPS_PER_DAY:
                break
        if not kept:
            continue
        positive=np.concatenate([g[0] for g in kept])
        comparison=np.concatenate([g[1] for g in kept])
        indices=np.r_[positive,comparison]
        batch=patches(image,data["candidate_flat"],indices)
        batch,_=mask_target_history(batch)
        scores,anchor=differentiable_scores(
            batch,teacher,reference,report_model,indices,lookup,rasters,
            transform,route_norm,device)
        bag_losses=[]
        start=0
        for j,(members,_,weight,_) in enumerate(kept):
            n=len(members)
            site=(torch.logsumexp(scores[start:start+n]/BAG_TEMPERATURE,0)-
                  math.log(n))*BAG_TEMPERATURE
            start+=n
            negative=scores[len(positive)+2*j:len(positive)+2*j+2]
            bag_losses.append(weight*F.softplus(negative-site).mean())
        loss=torch.stack(bag_losses).sum()/sum(g[2] for g in kept)+\
             ANCHOR_WEIGHT*anchor
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in teacher.parameters()
                                        if p.requires_grad],1.)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        accepted+=1
        if accepted%25==0:
            print("adapt steps",accepted,"loss",round(float(np.mean(losses[-25:])),5),
                  "anchor",round(float(anchor.detach().cpu()),5),flush=True)
    teacher.eval()
    return {"available_training_days":len(days),"selected_days":len(selected),
            "optimizer_steps":accepted,"mean_loss":float(np.mean(losses)),
            "last_20_loss":float(np.mean(losses[-20:])),
            "trainable_parameter_names":[n for n,p in teacher.named_parameters()
                                         if p.requires_grad]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold",type=int,choices=range(1,6))
    parser.add_argument("--folds",type=int,nargs="+",choices=range(1,6))
    parser.add_argument("--max-days",type=int,default=MAX_DAYS)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    data=load_grid()
    with np.load(FEATURES_NPZ) as saved:
        raw=saved["features"]
        coverage=saved["coverage"]
    previous=json.loads(ROUTE_JSON.read_text())
    previous_states=torch.load(ROUTE_MODEL,map_location="cpu",weights_only=True)["states"]
    rasters=load_sources(data)
    lookup,eligible=crop_index(data,coverage,rasters)
    elevation,_,_,_=terrain(data)
    district=data["district_id"].ravel()[data["candidate_flat"]]
    covered=coverage.ravel()[data["candidate_flat"]]
    folds=np.array_split(np.random.default_rng(917).permutation(50),5)
    device="mps" if torch.backends.mps.is_available() else "cpu"
    records,states=[],{}
    for fold_id,heldout in enumerate(folds,1):
        if args.fold is not None and fold_id!=args.fold:
            continue
        if args.folds is not None and fold_id not in args.folds:
            continue
        source=~np.isin(district,heldout)&covered
        target=np.isin(district,heldout)&covered
        source_grid=np.zeros(SHAPE,bool)
        source_grid.ravel()[data["candidate_flat"]]=source
        static,_=lidar_static(data,source,elevation)
        physics,_,transform=physics_for_fold(raw,coverage,source_grid,
                                             return_transform=True)
        route_proxy=incoming_proxy(raw,coverage)
        route,route_norm=normalize_proxy(route_proxy,source_grid,coverage)
        reference=FlowMaskCNN().to(device).eval()
        reference.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",
                                             weights_only=True)["state"])
        for p in reference.parameters():
            p.requires_grad_(False)
        teacher=FlowMaskCNN().to(device)
        teacher.load_state_dict(reference.state_dict())
        report_model=RoutedReportCNN().to(device).eval()
        report_model.load_state_dict(previous_states[fold_id][
            "spatial_with_incoming_route"])
        fit=finetune(data,source,static,physics,route,teacher,reference,
                     report_model,lookup,eligible,rasters,transform,route_norm,
                     SEED+fold_id*1000,args.max_days,device)
        # Rebuild every covered cell with the adapted final five-minute step.
        adapted_raw,adapted_coverage,_=build_features(data,teacher,device,reference)
        if not np.array_equal(adapted_coverage,coverage):
            raise ValueError("Fine-tuned physics coverage changed")
        adapted_physics=apply_physics_transform(adapted_raw,coverage,transform)
        adapted_route_proxy=incoming_proxy(adapted_raw,coverage)
        adapted_route=np.zeros_like(adapted_route_proxy)
        for channel in range(2):
            adapted_route[channel,coverage]=np.clip((
                adapted_route_proxy[channel,coverage]-route_norm["mean"][channel])/
                route_norm["std"][channel],-5,5)
        context_fn=make_context(np.concatenate((adapted_physics,adapted_route)))
        report_model=report_model.cpu()
        valid=score_days(data,source,static,report_model,
                         range(VALID_START,TEST_START),context_fn=context_fn)
        alpha=max(ALPHAS,key=lambda a:(valid[str(a)]["isolated_group_capture"],
                                       valid[str(a)]["isolated_cell_capture"],-a))
        held=score_days(data,target,static,report_model,
                        range(TEST_START,N_DAYS),context_fn=context_fn)
        old=next(f for f in previous["folds"] if f["fold"]==fold_id)["arms"][
            "spatial_with_incoming_route"]
        record={"fold":fold_id,"source_cells":int(source.sum()),
                "source_cells_with_full_10m_crop":int((source&eligible).sum()),
                "fit":fit,"source_validation":valid,"selected_alpha":alpha,
                "heldout_q4":held,"frozen_comparator_alpha":old["selected_alpha"],
                "frozen_comparator_heldout_q4":old["heldout_q4"]}
        records.append(record)
        states[fold_id]={k:v.cpu() for k,v in teacher.state_dict().items()}
        print("fold",fold_id,"frozen",old["heldout_q4"][str(old["selected_alpha"])][
            "isolated_hits"],"adapted",held[str(alpha)]["isolated_hits"],
            "alpha",alpha,flush=True)
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question":"Does last-layer 10 m SFINCS fine-tuning on masked report ranking beat the same frozen teacher and report CNN?",
            "max_days":args.max_days,"checkpoint":str(CHECKPOINT),
            "report_head":str(ROUTE_MODEL),
            "trainable":"10 m layers 8 and 9 plus depth and velocity heads; report CNN frozen",
            "gradient":"Masked 250 m group ranking -> 250 m spatial report CNN -> central 3x3 pooled 10 m depth, velocity, hidden features and incoming-flow cue -> trainable final 10 m layers",
            "first_two_steps":"Original frozen SFINCS model for both adaptation and rebuilt inference; only final step adapted",
            "training":"Up to 96 source-district 2023-24 days with isolated positives, three groups/day, same target history mask and group ranking loss, small output anchor to original checkpoint",
            "selection":"Jan-Sep 2025 source-only alpha, Q4 excluded districts held out",
            "folds":records,
            "limitations":["Earlier Q4 results informed this follow-up, so Q4 is exploratory",
                           "10 m crop-based differentiation approximates tiled citywide feature extraction",
                           "Only 9,808 candidate cells have qualifying terrain coverage",
                           "Synthetic scenario and unknown current water, not verified wetness",
                           "Unreported comparison cells are unknown, not verified dry"]}
    suffix=f"_fold_{args.fold}" if args.fold else (
        "_folds_"+"_".join(str(x) for x in args.folds) if args.folds else "")
    OUT_JSON.with_name(OUT_JSON.stem+suffix+".json").write_text(
        json.dumps(result,indent=2)+"\n")
    torch.save({"states":states},OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt"))
    print("saved",OUT_JSON.with_name(OUT_JSON.stem+suffix+".json"),flush=True)


if __name__=="__main__":
    main()
