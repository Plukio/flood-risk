"""Test an explicit one-cell incoming-flow cue beside the spatial report CNN.

This proxy routes pooled depth times signed velocity to four neighboring
250 m cells. It is not a drainage-network flux or a calibrated 250 m solver.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from flood_citywide_history_cnn_transfer import SEED
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import ALPHAS, ContextResidualCNN, pool, score_days
from flood_citywide_isolated_group_mask import mask_self_check, train
from flood_citywide_route_pretrain import lidar_static, terrain
from flood_multi_target_citywide_data import OUT
from flood_sfincs_report_features import OUT_NPZ as FEATURES_NPZ
from flood_sfincs_report_mask_transfer import ARM, physics_for_fold, make_context

OUT_JSON=OUT/"q4_2025_sfincs_directed_neighbor_mask_fivefold.json"
OUT_MODEL=OUT/"q4_2025_sfincs_directed_neighbor_mask_fivefold.pt"
ARMS=("spatial_no_route", "spatial_with_incoming_route")


class RoutedReportCNN(ContextResidualCNN):
    def __init__(self):
        super().__init__()
        self.entry=nn.Conv2d(21+16,8,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


def incoming_proxy(raw:np.ndarray,coverage:np.ndarray):
    """Orient one-hop messages by velocity sign on the 250 m grid."""
    depth=np.maximum(raw[0],0)*.001  # pooled predicted mm -> m
    east=depth*raw[2]*coverage
    north=depth*raw[3]*coverage
    incoming=np.zeros(SHAPE,np.float32)
    # Positive eastward u from the west, negative u from the east.
    incoming[:,1:]+=np.maximum(east[:,:-1],0)
    incoming[:,:-1]+=np.maximum(-east[:,1:],0)
    # Grid row increases south; positive northward v comes from row+1.
    incoming[:-1,:]+=np.maximum(north[1:,:],0)
    incoming[1:,:]+=np.maximum(-north[:-1,:],0)
    outgoing=np.abs(east)+np.abs(north)
    incoming*=coverage
    net=(incoming-outgoing)*coverage
    return np.stack((incoming,net)).astype(np.float32)


def normalize_proxy(proxy:np.ndarray,source_grid:np.ndarray,coverage:np.ndarray):
    scope=source_grid&coverage
    mean=proxy[:,scope].mean(axis=1)
    std=np.maximum(proxy[:,scope].std(axis=1),1e-5)
    result=np.zeros_like(proxy)
    result[:,coverage]=((proxy[:,coverage]-mean[:,None])/std[:,None]).clip(-5,5)
    return result,{"mean":mean.tolist(),"std":std.tolist()}


def context_with_extra(extra:np.ndarray):
    # Reuse the exact history order and target-mask indices from the first test.
    return make_context(extra)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed-offset",type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(4)
    data=load_grid()
    with np.load(FEATURES_NPZ) as saved:
        raw=saved["features"]
        coverage=saved["coverage"]
        if not np.array_equal(saved["candidate_flat"],data["candidate_flat"]):
            raise ValueError("Physics and report grids differ")
    proxy=incoming_proxy(raw,coverage)
    if not np.all(proxy[0]>=0) or not np.isfinite(proxy).all():
        raise ValueError("Invalid directed proxy")
    elevation,_,_,_=terrain(data)
    district=data["district_id"].ravel()[data["candidate_flat"]]
    covered=coverage.ravel()[data["candidate_flat"]]
    folds=np.array_split(np.random.default_rng(917).permutation(50),5)
    records,states=[],{}
    for fold_id,heldout in enumerate(folds,1):
        source=~np.isin(district,heldout)&covered
        target=np.isin(district,heldout)&covered
        source_grid=np.zeros(SHAPE,bool)
        source_grid.ravel()[data["candidate_flat"]]=source
        static,_=lidar_static(data,source,elevation)
        mask_self_check(data,static)
        physics,_=physics_for_fold(raw,coverage,source_grid)
        route,route_norm=normalize_proxy(proxy,source_grid,coverage)
        record={"fold":fold_id,"source_cells":int(source.sum()),
                "target_cells":int(target.sum()),"route_normalization":route_norm,
                "arms":{}}
        states[fold_id]={}
        for arm in ARMS:
            candidate=np.concatenate((physics,route if arm==ARMS[1] else
                                      np.zeros_like(route)),axis=0)
            context_fn=context_with_extra(candidate)
            model,fit=train(data,source,static,SEED+fold_id*1000+args.seed_offset,ARM,
                            model_factory=RoutedReportCNN,context_fn=context_fn)
            valid=score_days(data,source,static,model,
                             range(VALID_START,TEST_START),context_fn=context_fn)
            alpha=max(ALPHAS,key=lambda a:(valid[str(a)]["isolated_group_capture"],
                                           valid[str(a)]["isolated_cell_capture"],-a))
            held=score_days(data,target,static,model,
                            range(TEST_START,N_DAYS),context_fn=context_fn)
            record["arms"][arm]={"fit":fit,"source_validation":valid,
                                 "selected_alpha":alpha,"heldout_q4":held}
            states[fold_id][arm]={k:v.cpu() for k,v in model.state_dict().items()}
            print("fold",fold_id,arm,"alpha",alpha,"Q4 cells",
                  held[str(alpha)]["isolated_hits"],"groups",
                  held[str(alpha)]["group_hits"],flush=True)
        records.append(record)
    pooled={}
    for arm in ARMS:
        pooled[arm]={str(a):pool([f["arms"][arm]["heldout_q4"][str(a)]
                                  for f in records]) for a in ALPHAS}
        pooled[arm]["source_selected"]=pool([
            f["arms"][arm]["heldout_q4"][str(f["arms"][arm]["selected_alpha"])]
            for f in records])
    if pooled[ARMS[0]]["0.0"]["isolated_positive"]!=149 or\
       pooled[ARMS[1]]["0.0"]["isolated_positive"]!=149:
        raise ValueError("Frozen covered queue changed")
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed_offset":args.seed_offset,
            "question":"Does an explicitly oriented one-hop incoming-flow proxy help the frozen-physics spatial CNN?",
            "teacher_features":str(FEATURES_NPZ),
            "incoming_rule":"250 m pooled simulated depth [m] times signed velocity [m/s] is shifted one cell in the vector's cardinal direction; negative and positive components route to opposite neighbors. Also supply incoming minus outgoing.",
            "architecture":"Both arms use the same 37-input, eight-hidden-channel 250 m CNN and identical masked group ranking training. The control's two route channels are zero.",
            "selection":"Source-district Jan-Sep 2025 validation only; Q4 excluded districts for report evaluation",
            "queue":"Daily top 10% among 9,808 covered cells, 20% reserved for strict isolated candidates",
            "folds":records,"pooled_heldout_q4":pooled,
            "limitations":["Pooled center velocity is not a measured edge flux or verified drainage connection",
                           "Fixed synthetic rain, no observed initial water, drains or pumps",
                           "Unreported comparisons are unknown, not verified dry",
                           "Q4 report holdout has been repeatedly inspected"]}
    suffix=f"_seed_{args.seed_offset}" if args.seed_offset else ""
    OUT_JSON.with_name(OUT_JSON.stem+suffix+".json").write_text(
        json.dumps(result,indent=2)+"\n")
    torch.save({"states":states},OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt"))
    print("pooled",{k:(v["source_selected"]["isolated_hits"],
                        v["source_selected"]["group_hits"])
                    for k,v in pooled.items()},flush=True)


if __name__=="__main__":
    main()
