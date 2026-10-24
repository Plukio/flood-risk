"""Evaluate matched joint-CNN mask arms on urban and satellite holdouts."""
from __future__ import annotations

import json
import math
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import convolve, gaussian_filter, label

from flood_citywide_target_site_mask_bma_transfer import located_bma
from flood_citywide_history_ranker import N_DAYS, TEST_START, load_grid
from flood_citywide_spatial_novel_queue import queue_weights, top_weights
from flood_joint_target_site_mask_train import (
    ARMS, CONTROL_ARMS, DEST, HOLDOUT_REGIONS,
)
from flood_label_audit import scene_path, validate_scene
from flood_multi_target_citywide_eval import (
    Q4_DAYS, build_arrays, paired_interval,
)
from flood_multi_target_train import MultiTargetCNN, normalize
from flood_open_weak_data import FEATURE_NAMES, OUTPUT

OUT_JSON=DEST/"holdout_evaluation.json"
QUOTA=.20
ALL_MODEL_ARMS=(*ARMS,*CONTROL_ARMS)
METHODS=("fixed_full","simple_neighbor",*ALL_MODEL_ARMS)


def load_models() -> tuple[dict,dict]:
    models={}
    norm=None
    seed=None
    for arm in ALL_MODEL_ARMS:
        state=torch.load(DEST/f"{arm}.pt",weights_only=False)
        if norm is None:
            norm=state["normalization"]
            seed=state["seed"]
        elif norm!=state["normalization"]:
            raise ValueError("Matched arms use different normalization")
        elif seed!=state["seed"]:
            raise ValueError("Matched arms use different seeds")
        if set(state["excluded_satellite_regions"])!=HOLDOUT_REGIONS:
            raise ValueError("Satellite holdout changed")
        model=MultiTargetCNN().eval()
        model.load_state_dict(state["state_dict"])
        models[arm]=model
    return models,norm


@torch.no_grad()
def urban_scores(models:dict,norm:dict,records:list[dict],shape:tuple[int,int]) -> dict:
    output={name:np.zeros((len(Q4_DAYS),*shape),np.float32) for name in ALL_MODEL_ARMS}
    labels=np.zeros((len(Q4_DAYS),*shape),np.uint8)
    candidate=np.zeros(shape,bool)
    new=np.zeros(shape,bool)
    for item in records:
        name=item["name"]
        parts=name.split("_")
        row,col=int(parts[1][1:]),int(parts[2][1:])
        rr=slice((3-1-row)*80,(3-row)*80)
        cc=slice(col*80,(col+1)*80)
        labels[:,rr,cc]=item["truth"]
        candidate[rr,cc]=item["candidate"]
        new[rr,cc]=item["new"]
        static,weather=normalize(item["static"],item["weather"],norm)
        for arm,model in models.items():
            days=[]
            for start in range(0,len(Q4_DAYS),8):
                stop=min(start+8,len(Q4_DAYS))
                prior=np.log1p(item["history"][start:stop].astype(np.float32))/math.log1p(20)
                history=np.stack((prior,np.ones_like(prior)),axis=1)
                batch_static=np.broadcast_to(static,(stop-start,*static.shape)).copy()
                predicted=model(torch.from_numpy(batch_static),
                                torch.from_numpy(weather[start:stop]),
                                torch.from_numpy(history))["traffy"].numpy()
                days.append(predicted)
            output[arm][:,rr,cc]=np.concatenate(days)
        print("urban scored",name,flush=True)
    return {"scores":output,"labels":labels,"candidate":candidate,"new":new}


def empty_row() -> dict:
    n=len(Q4_DAYS)
    return {"positive_cell_days":0,"captured":0.,
            "novel_positive_cell_days":0,"novel_captured":0.,
            "isolated_positive_cell_days":0,"isolated_captured":0.,
            "daily_positive":[0]*n,"daily_captured":[0.]*n,
            "daily_novel_positive":[0]*n,"daily_novel_captured":[0.]*n,
            "daily_isolated_positive":[0]*n,"daily_isolated_captured":[0.]*n,
            "isolated_hit_sites":[]}


def urban_metrics(data:dict, scored:dict) -> dict:
    if not np.array_equal(scored["candidate"],data["candidate"]):
        raise ValueError("Urban candidate grid changed")
    if not np.array_equal(scored["new"],data["new"]):
        raise ValueError("Urban geography exclusion changed")
    if not np.array_equal(scored["labels"],data["reports"][TEST_START:]):
        raise ValueError("Urban report outcomes do not align")
    kernel=np.ones((3,3),np.uint8)
    kernel[1,1]=0
    scopes={"all_citywide":data["candidate"],
            "outside_urban_training_patches":data["new"]}
    result={scope:{name:empty_row() for name in METHODS} for scope in scopes}
    for offset,day in enumerate(range(TEST_START,N_DAYS)):
        prior365=(data["prefix"][day]-data["prefix"][day-365]).astype(np.float32)
        prior90=(data["prefix"][day]-data["prefix"][day-90]).astype(np.float32)
        near=gaussian_filter(prior365,1,mode="constant")
        middle=gaussian_filter(prior365,3,mode="constant")
        wide=gaussian_filter(prior365,6,mode="constant")
        base=np.log1p(prior365)+.4*np.log1p(prior90)+.4*np.log1p(near)
        simple=.6*np.log1p(near)+.3*np.log1p(middle)+.1*np.log1p(wide)
        novel=prior365==0
        adjacent=convolve((prior365>0).astype(np.uint8),kernel,
                          mode="constant")>0
        truth=scored["labels"][offset]>0
        for scope_name,scope in scopes.items():
            b,s,n,y=base[scope],simple[scope],novel[scope],truth[scope]
            isolated=n & ~adjacent[scope]
            picks={"fixed_full":top_weights(b,.1*len(b)),
                   "simple_neighbor":queue_weights(b,s,n,QUOTA)}
            for arm in ALL_MODEL_ARMS:
                picks[arm]=queue_weights(b,scored["scores"][arm][offset,scope],
                                        n,QUOTA)
            for name,w in picks.items():
                row=result[scope_name][name]
                novel_positive=n & y
                isolated_positive=isolated & y
                row["positive_cell_days"]+=int(y.sum())
                row["captured"]+=float(np.dot(w,y))
                row["novel_positive_cell_days"]+=int(novel_positive.sum())
                row["novel_captured"]+=float(np.dot(w,novel_positive))
                row["isolated_positive_cell_days"]+=int(isolated_positive.sum())
                row["isolated_captured"]+=float(np.dot(w,isolated_positive))
                row["daily_positive"][offset]=int(y.sum())
                row["daily_captured"][offset]=float(np.dot(w,y))
                row["daily_novel_positive"][offset]=int(novel_positive.sum())
                row["daily_novel_captured"][offset]=float(np.dot(w,novel_positive))
                row["daily_isolated_positive"][offset]=int(isolated_positive.sum())
                row["daily_isolated_captured"][offset]=float(np.dot(w,isolated_positive))
                if name in ALL_MODEL_ARMS and np.any(isolated_positive & (w>0)):
                    global_flats=np.flatnonzero(scope)[isolated_positive & (w>0)]
                    weights=w[isolated_positive & (w>0)]
                    for global_flat,weight in zip(global_flats,weights):
                        grid_row,grid_col=divmod(int(global_flat),data["candidate"].shape[1])
                        code=data["districts"][int(data["district_id"][grid_row,grid_col])]["code"]
                        row["isolated_hit_sites"].append({
                            "date":str(Q4_DAYS[offset]),"district_code":code,
                            "grid_row":grid_row,"grid_col":grid_col,
                            "selection_weight":round(float(weight),5)})
    for scope_name,rows in result.items():
        for name,row in rows.items():
            row["overall_capture"]=round(row["captured"]/row["positive_cell_days"],5)
            row["novel_capture"]=round(row["novel_captured"]/row["novel_positive_cell_days"],5)
            row["isolated_capture"]=round(row["isolated_captured"]/row["isolated_positive_cell_days"],5)
            for key in ("captured","novel_captured","isolated_captured"):
                row[key]=round(row[key],4)
            if name in ALL_MODEL_ARMS:
                sites=row["isolated_hit_sites"]
                row["isolated_hit_districts"]=len({s["district_code"] for s in sites})
                row["isolated_hit_dates"]=len({s["date"] for s in sites})
                row["isolated_distinct_cells"]=len({
                    (s["grid_row"],s["grid_col"]) for s in sites})
                clusters=0
                for event_day in {s["date"] for s in sites}:
                    selected=np.zeros(data["candidate"].shape,bool)
                    for site in sites:
                        if site["date"]==event_day:
                            selected[site["grid_row"],site["grid_col"]]=True
                    clusters+=label(selected,structure=np.ones((3,3),bool))[1]
                row["isolated_hit_adjacent_clusters"]=clusters
        print("urban",scope_name,{name:(row["overall_capture"],
                                        row["novel_capture"],row["isolated_capture"])
                                  for name,row in rows.items()},flush=True)
    if result["all_citywide"]["fixed_full"]["positive_cell_days"]!=7209:
        raise ValueError("Q4 citywide report positives changed")
    if (result["all_citywide"]["fixed_full"]["novel_positive_cell_days"]!=1664
        or result["all_citywide"]["fixed_full"]["isolated_positive_cell_days"]!=332
        or result["outside_urban_training_patches"]["fixed_full"]["positive_cell_days"]!=3605
        or result["outside_urban_training_patches"]["fixed_full"]["novel_positive_cell_days"]!=959
        or result["outside_urban_training_patches"]["fixed_full"]["isolated_positive_cell_days"]!=220):
        raise ValueError("Q4 novel/isolated denominators changed")
    return result


def bma_metrics(data:dict,scored:dict) -> dict:
    records,audit=located_bma(data)
    if len(records)!=89:
        raise ValueError("BMA aligned-site count changed")
    by_day={}
    for record in records:
        by_day.setdefault(record["day"],[]).append(record)
    result={name:{"site_days":0,"depth_cm_sum":0.,
                  "captured":{method:0. for method in METHODS},
                  "captured_depth_cm":{method:0. for method in METHODS}}
            for name in ("all","repeat","novel_adjacent","novel_isolated")}
    candidate=data["candidate"]
    kernel=np.ones((3,3),np.uint8)
    kernel[1,1]=0
    for day,sites in by_day.items():
        offset=day-TEST_START
        prior365=(data["prefix"][day]-data["prefix"][day-365]).astype(np.float32)
        prior90=(data["prefix"][day]-data["prefix"][day-90]).astype(np.float32)
        near=gaussian_filter(prior365,1,mode="constant")
        middle=gaussian_filter(prior365,3,mode="constant")
        wide=gaussian_filter(prior365,6,mode="constant")
        base=np.log1p(prior365)+.4*np.log1p(prior90)+.4*np.log1p(near)
        simple=.6*np.log1p(near)+.3*np.log1p(middle)+.1*np.log1p(wide)
        novel=prior365==0
        adjacent=convolve((prior365>0).astype(np.uint8),kernel,
                          mode="constant")>0
        b,s,n=base[candidate],simple[candidate],novel[candidate]
        picks={"fixed_full":top_weights(b,.1*len(b)),
               "simple_neighbor":queue_weights(b,s,n,QUOTA)}
        for arm in ALL_MODEL_ARMS:
            picks[arm]=queue_weights(b,scored["scores"][arm][offset,candidate],
                                    n,QUOTA)
        for site in sites:
            i=site["candidate_index"]
            flat=data["candidate_flat"][i]
            grid_row,grid_col=divmod(int(flat),candidate.shape[1])
            group=("repeat" if not novel[grid_row,grid_col] else
                   "novel_adjacent" if adjacent[grid_row,grid_col]
                   else "novel_isolated")
            for name in ("all",group):
                row=result[name]
                row["site_days"]+=1
                row["depth_cm_sum"]+=site["depth_cm"]
                for method,w in picks.items():
                    row["captured"][method]+=float(w[i])
                    row["captured_depth_cm"][method]+=float(w[i])*site["depth_cm"]
    for row in result.values():
        row["capture"]={method:round(value/row["site_days"],5)
                        if row["site_days"] else None
                        for method,value in row["captured"].items()}
        row["depth_weighted_capture"]={method:round(value/row["depth_cm_sum"],5)
                                       if row["depth_cm_sum"] else None
                                       for method,value in row["captured_depth_cm"].items()}
        row["depth_cm_sum"]=round(row["depth_cm_sum"],2)
        row["captured"]={method:round(value,3)
                         for method,value in row["captured"].items()}
        row["captured_depth_cm"]={method:round(value,2)
                                  for method,value in row["captured_depth_cm"].items()}
    print("BMA",{name:(row["site_days"],row["capture"])
                 for name,row in result.items()},flush=True)
    return {"location_audit":audit,"groups":result}


@torch.no_grad()
def satellite_metrics(models:dict,norm:dict) -> dict:
    manifest=json.loads((OUTPUT/"data_audit.json").read_text())
    if tuple(manifest["features"])!=FEATURE_NAMES:
        raise ValueError("Satellite feature order changed")
    scenes=[]
    for entry in manifest["scenes"]:
        if entry["region"] not in HOLDOUT_REGIONS:
            continue
        with np.load(scene_path(entry),allow_pickle=False) as z:
            raw={key:z[key].astype(np.float32) for key in
                 ("static","weather","wet","observed")}
        validate_scene(entry,raw,manifest)
        if float(raw["wet"].sum())<=1:
            continue
        static,weather=normalize(raw["static"],raw["weather"],norm)
        empty=torch.zeros((1,2,80,80),dtype=torch.float32)
        predictions={}
        for arm in ARMS:
            model=models[arm]
            prediction=model(torch.from_numpy(static[None]),
                             torch.from_numpy(weather[None]),empty)
            head="gistda" if entry["source"]=="gistda2026" else (
                 "eos" if entry["source"]=="eos2026" else "unosat")
            predictions[arm]=prediction[head][0].numpy()
        predictions["crop_cover"]=raw["static"][FEATURE_NAMES.index(
            "worldcover_crop_fraction_2021")]
        predictions["built_cover"]=raw["static"][FEATURE_NAMES.index(
            "worldcover_built_fraction_2021")]
        area=float(raw["wet"].sum())
        captured={name:float(np.dot(top_weights(score.ravel(),640),
                                   raw["wet"].ravel()))
                  for name,score in predictions.items()}
        scenes.append({"source":entry["source"],"region":entry["region"],
                       "date":entry["day"],"mapped_positive_area_units":round(area,5),
                       "top_10pct_capture":{name:round(value/area,5)
                                            for name,value in captured.items()},
                       "captured_area_units":captured})
    groups={}
    for label,selected in (("pathum_thani_gistda_8_dates",
                            [s for s in scenes if s["source"]=="gistda2026"]),
                           ("bangkok_eos_2_patches",
                            [s for s in scenes if s["source"]=="eos2026"]),
                           ("pathum_thani_unosat_2_wet_dates",
                            [s for s in scenes if s["source"]=="unosat2021"])):
        denom=sum(s["mapped_positive_area_units"] for s in selected)
        groups[label]={"scenes":len(selected),
                       "mapped_positive_area_units":round(denom,5),
                       "top_10pct_capture":{name:round(
                           sum(s["captured_area_units"][name] for s in selected)/denom,5)
                           for name in (*ARMS,"crop_cover","built_cover")}}
    if (groups["pathum_thani_gistda_8_dates"]["scenes"]!=8 or
        groups["bangkok_eos_2_patches"]["scenes"]!=2 or
        groups["pathum_thani_unosat_2_wet_dates"]["scenes"]!=2):
        raise ValueError("Satellite holdout scene counts changed")
    print("satellite",groups,flush=True)
    return {"groups":groups,"scenes":scenes}


def interval_for(row:dict,kind:str) -> dict:
    if kind=="overall":
        return {"positive_cell_days":row["positive_cell_days"],
                "daily_positive":row["daily_positive"],
                "daily_captured":row["daily_captured"]}
    return {"positive_cell_days":row[f"{kind}_positive_cell_days"],
            "daily_positive":row[f"daily_{kind}_positive"],
            "daily_captured":row[f"daily_{kind}_captured"]}


def main(checkpoint_dir:Path=DEST,joint_only:bool=False) -> None:
    global DEST,OUT_JSON,ALL_MODEL_ARMS,METHODS
    DEST=checkpoint_dir.resolve()
    OUT_JSON=DEST/"holdout_evaluation.json"
    ALL_MODEL_ARMS=ARMS if joint_only else (*ARMS,*CONTROL_ARMS)
    METHODS=("fixed_full","simple_neighbor",*ALL_MODEL_ARMS)
    torch.set_num_threads(4)
    models,norm=load_models()
    records,_,_=build_arrays()
    data=load_grid()
    scored=urban_scores(models,norm,records,data["candidate"].shape)
    urban=urban_metrics(data,scored)
    bma=bma_metrics(data,scored)
    satellite=satellite_metrics(models,norm)
    intervals={}
    comparisons=(("target_centered_3x3","random_block_matched"),
                 ("target_centered_3x3","urban_only_target_centered_3x3"),
                 ("random_block_matched","urban_only_random_block_matched"),
                 ("urban_only_target_centered_3x3",
                  "urban_only_random_block_matched"))
    for scope,rows in urban.items():
        for kind in ("overall","novel","isolated"):
            for first,second in comparisons:
                if first not in rows or second not in rows:
                    continue
                a=interval_for(rows[first],kind)
                b=interval_for(rows[second],kind)
                intervals[f"{scope}:{kind}:{first}_minus_{second}"]=paired_interval(a,b)
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question":"Do target-centered urban-history masks help a joint satellite/urban CNN transfer?",
            "training":"2022-2024 Traffy/BMA urban sites plus external satellite scenes",
            "urban_test":"Q4 2025 Bangkok report-positive cell-days; held-out time, geography outside two urban patches",
            "satellite_holdouts":sorted(HOLDOUT_REGIONS),
            "urban_queue":"Daily top 10% cells; 20% of slots reserved for zero-own-365-day-history cells",
            "urban":urban,"bma_positive_sites":bma,
            "satellite_positive_area":satellite,
            "paired_weekly_date_block_intervals":intervals,
            "limitations":["Unreported urban cells and transparent satellite pixels are unknown, not verified dry",
                           "Satellite 2026 training makes Q4 2025 a retrospective diagnostic",
                           "Satellite heads do not read report-history masks directly; their change is only through the shared encoder",
                           "Full-day rain may occur after reports or satellite acquisitions",
                           "Q4 2025 labels have been inspected in earlier experiments"]}
    OUT_JSON.write_text(json.dumps(result,indent=2)+"\n")
    print("saved",OUT_JSON,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir",type=Path,default=DEST)
    parser.add_argument("--joint-only",action="store_true")
    args=parser.parse_args()
    main(args.checkpoint_dir,args.joint_only)
