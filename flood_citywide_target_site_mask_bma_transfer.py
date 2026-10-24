"""Frozen five-fold novel-site CNN queues on independent BMA road-flood sites.

Q4 2025 BMA positives are conservatively located through exact name matches
to BMA's 2026 risk inventory. No BMA data fit or select the Traffy-trained
CNNs. Missing BMA reports are unknown, not dry road observations.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
import torch
from scipy.ndimage import convolve

from flood_bangkok_multitask_depth_cnn import matched_depths
from flood_bangkok_transfer import load_districts
from flood_bangkapi_surface_grid import TO_UTM
from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, load_grid,
)
from flood_citywide_spatial_novel_queue import (
    SpatialNovelCNN, day_maps, queue_weights, static_image, top_weights,
)
from flood_citywide_target_site_mask_test import (
    ARMS, OUT_JSON, OUT_MODEL, QUOTA,
)
from flood_multi_target_citywide_data import OUT, ROWS, TILE_M, X0, Y0
from flood_multi_target_citywide_eval import paired_interval
from flood_multi_target_data import START

DEST = OUT / "q4_2025_target_site_mask_bma_transfer.json"
METHODS = ("fixed_full", "simple_neighbor", *ARMS,
           "source_selected_location")
GROUPS = ("all", "repeat", "novel_adjacent", "novel_isolated")


def located_bma(data: dict) -> tuple[list[dict], dict]:
    district_map = {d["code"]:d for d in load_districts()}
    _, matched, source_audit = matched_depths(district_map)
    raw = [r for r in matched if "2025-10-01" <= r["date"] <= "2025-12-31"]
    flat = data["candidate_flat"]
    candidate = data["candidate"]
    district_id=data["district_id"]
    counters=defaultdict(int)
    by_cell_day={}
    for r in raw:
        x,y=TO_UTM.transform(r["inventory_lon"],r["inventory_lat"])
        row=math.floor((Y0+ROWS*TILE_M-y)/250)
        col=math.floor((x-X0)/250)
        if not (0 <= row < SHAPE[0] and 0 <= col < SHAPE[1]):
            counters["outside_global_tiles"]+=1
            continue
        if not candidate[row,col]:
            counters["outside_candidate_mask"]+=1
            continue
        assigned=data["districts"][int(district_id[row,col])]["code"]
        if assigned!=r["district_code"]:
            counters["district_assignment_mismatch"]+=1
            continue
        global_flat=row*SHAPE[1]+col
        candidate_index=int(np.searchsorted(flat,global_flat))
        if candidate_index>=len(flat) or flat[candidate_index]!=global_flat:
            raise ValueError("Candidate index mapping failed")
        day=(date.fromisoformat(r["date"])-START).days
        key=(day,candidate_index)
        if key in by_cell_day:
            by_cell_day[key]["depth_cm"]=max(
                by_cell_day[key]["depth_cm"],float(r["depth_cm"]))
            by_cell_day[key]["matched_rows"]+=1
        else:
            by_cell_day[key]={"day":day,"date":r["date"],
                              "candidate_index":candidate_index,
                              "district_code":assigned,
                              "depth_cm":float(r["depth_cm"]),
                              "matched_rows":1}
    rows=sorted(by_cell_day.values(),key=lambda r:(r["day"],r["candidate_index"]))
    counters["q4_exact_matched_rows"]=len(raw)
    counters["aligned_unique_cell_days"]=len(rows)
    return rows,{"source_audit":source_audit,"alignment":dict(counters),
                 "location_method":"exact normalized BMA road-area name to 2026 risk inventory point"}


def empty_group() -> dict:
    return {"positive_cell_days":0,"depth_cm_sum":0.,
            "daily_positive":[0]*(N_DAYS-TEST_START),
            "daily_depth":[0.]*(N_DAYS-TEST_START),
            "methods":{name:{"captured":0.,"captured_depth_cm":0.,
                             "daily_captured":[0.]*(N_DAYS-TEST_START),
                             "daily_captured_depth":[0.]*(N_DAYS-TEST_START)}
                       for name in METHODS}}


def main() -> None:
    torch.set_num_threads(4)
    data=load_grid()
    records,audit=located_bma(data)
    if not records:
        raise ValueError("No aligned BMA site-days")
    experiment=json.loads(OUT_JSON.read_text())
    states=torch.load(OUT_MODEL,weights_only=True)["states"]
    district=data["district_id"].ravel()[data["candidate_flat"]]
    fold_by_code={code:f["fold"] for f in experiment["folds"]
                  for code in f["district_codes"]}
    for r in records:
        r["fold"]=fold_by_code[r["district_code"]]
    by_fold_day=defaultdict(list)
    for r in records:
        by_fold_day[(r["fold"],r["day"])].append(r)
    groups={name:empty_group() for name in GROUPS}
    per_fold=[]
    kernel=np.ones((3,3),np.uint8)
    kernel[1,1]=0
    for fold in experiment["folds"]:
        fold_number=fold["fold"]
        codes=set(fold["district_codes"])
        ids=np.asarray([i for i,d in enumerate(data["districts"])
                        if d["code"] in codes])
        target=np.isin(district,ids)
        source=~target
        scope=np.flatnonzero(target)
        local=np.full(len(district),-1,np.int32)
        local[scope]=np.arange(len(scope))
        static,_=static_image(data,source)
        models={}
        for arm in ARMS:
            model=SpatialNovelCNN().eval()
            model.load_state_dict(states[fold_number][arm])
            models[arm]=model
        fold_site_days=0
        for day in sorted(d for (f,d) in by_fold_day if f==fold_number):
            image,base,simple,novel=day_maps(data,day,static)
            prior=(data["prefix"][day]-data["prefix"][day-365]) > 0
            adjacent=(convolve(prior.astype(np.uint8),kernel,
                               mode="constant") > 0)[data["candidate"]]
            b=base[target]
            n=novel[target]
            picks={"fixed_full":top_weights(b,.1*len(b)),
                   "simple_neighbor":queue_weights(b,simple[target],n,QUOTA)}
            for arm,model in models.items():
                score=model.full_scores(image,data["candidate"])
                picks[arm]=queue_weights(b,score[target],n,QUOTA)
            picks["source_selected_location"]=picks[fold["source_selected_arm"]]
            for r in by_fold_day[(fold_number,day)]:
                i=r["candidate_index"]
                j=local[i]
                if j<0:
                    raise ValueError("BMA point outside its held-out queue")
                is_novel=bool(novel[i])
                is_adjacent=bool(adjacent[i])
                r["own_history_365d"]=not is_novel
                r["adjacent_history_365d"]=is_adjacent
                r["group"]=("repeat" if not is_novel else
                            "novel_adjacent" if is_adjacent else "novel_isolated")
                r["selection_weight"]={name:round(float(w[j]),5)
                                       for name,w in picks.items()}
                for group_name in ("all",r["group"]):
                    group=groups[group_name]
                    offset=day-TEST_START
                    group["positive_cell_days"]+=1
                    group["depth_cm_sum"]+=r["depth_cm"]
                    group["daily_positive"][offset]+=1
                    group["daily_depth"][offset]+=r["depth_cm"]
                    for name,w in picks.items():
                        weight=float(w[j])
                        method=group["methods"][name]
                        method["captured"]+=weight
                        method["captured_depth_cm"]+=weight*r["depth_cm"]
                        method["daily_captured"][offset]+=weight
                        method["daily_captured_depth"][offset]+=weight*r["depth_cm"]
                fold_site_days+=1
        per_fold.append({"fold":fold_number,"bma_site_days":fold_site_days,
                         "source_selected_arm":fold["source_selected_arm"]})
        print("fold",fold_number,"BMA site-days",fold_site_days,flush=True)
    if sum(f["bma_site_days"] for f in per_fold)!=len(records):
        raise ValueError("BMA site-days were lost or duplicated")
    summary={}
    for group_name,group in groups.items():
        denom=group["positive_cell_days"]
        depth=group["depth_cm_sum"]
        summary[group_name]={"positive_cell_days":denom,
                             "depth_cm_sum":round(depth,2),
                             "methods":{name:{
                                 "captured":round(row["captured"],3),
                                 "capture":round(row["captured"]/denom,5) if denom else None,
                                 "captured_depth_cm":round(row["captured_depth_cm"],3),
                                 "depth_weighted_capture":round(row["captured_depth_cm"]/depth,5) if depth else None}
                                 for name,row in group["methods"].items()}}
    intervals={}
    for group_name,group in groups.items():
        denom=group["positive_cell_days"]
        if not denom:
            continue
        for first,second in (("target_centered_3x3","random_block_matched"),
                             ("target_centered_3x3","simple_neighbor"),
                             ("source_selected_location","simple_neighbor")):
            a={"positive_cell_days":denom,"daily_positive":group["daily_positive"],
               "daily_captured":group["methods"][first]["daily_captured"]}
            b={"positive_cell_days":denom,"daily_positive":group["daily_positive"],
               "daily_captured":group["methods"][second]["daily_captured"]}
            intervals[f"{group_name}:{first}_minus_{second}"]=paired_interval(a,b)
    output={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target":"BMA recorded positive road-flood site-days with reported depth",
            "period":["2025-10-01","2025-12-31"],
            "queue":"Each held-out district fold selects top 10% of candidate 250m cells per day; novel reserve uses 20% of slots",
            "model_source":"Frozen Traffy-trained five-fold target-site mask models, no BMA fitting or selection",
            "bma_location_audit":audit,"per_fold":per_fold,
            "summary":summary,"paired_weekly_date_block_intervals":intervals,
            "site_days":records,
            "limitations":["No BMA dry-cell labels; positive-only capture, not precision or flood probability",
                           "BMA archive covers selected major roads, not all wet locations",
                           "2026 inventory anchors approximate 2025 event locations",
                           "Q4 2025 has been reused across experiments"]}
    DEST.write_text(json.dumps(output,indent=2)+"\n")
    print("summary",json.dumps(summary,indent=2),flush=True)
    print("saved",DEST,flush=True)


if __name__=="__main__":
    main()
