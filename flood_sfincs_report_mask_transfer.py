"""Frozen 10 m depth/direction features -> spatial 250 m masked report CNN.

The report network is convolutional across 250 m cells: neighbor depth,
signed east/north flow, and frozen 10 m trunk features can affect the target.
Only the report CNN is fitted to Traffy labels. Same group loss, masks, split,
queue and model width are used for all arms.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from flood_citywide_history_cnn_transfer import SEED
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import (
    ALPHAS, ContextResidualCNN, context, pool, score_days,
)
from flood_citywide_isolated_group_mask import mask_self_check, train
from flood_citywide_route_pretrain import lidar_static, terrain
from flood_multi_target_citywide_data import OUT
from flood_sfincs_report_features import OUT_NPZ as FEATURES_NPZ, OUT_JSON as FEATURES_JSON

OUT_JSON = OUT / "q4_2025_sfincs_frozen_spatial_report_mask_fivefold.json"
OUT_MODEL = OUT / "q4_2025_sfincs_frozen_spatial_report_mask_fivefold.pt"
ARM = "isolated_plus_masked_adjacent"
ARMS = ("history_cnn", "depth_and_hidden_no_vectors", "depth_direction_hidden")
EXTRA = 14  # 5 scenario summaries + 8 frozen-trunk PCs + coverage


class FrozenPhysicsReportCNN(ContextResidualCNN):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(21+EXTRA,8,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


def physics_for_fold(raw:np.ndarray,coverage:np.ndarray,source_grid:np.ndarray,
                     return_transform:bool=False):
    """Source-only scaling/PCA; missing cells stay zero and carry a mask."""
    fit_mask=coverage&source_grid
    if fit_mask.sum()<100:
        raise ValueError("Too few source physics cells")
    summaries=raw[:5]
    average=summaries[:,fit_mask].mean(axis=1)
    spread=np.maximum(summaries[:,fit_mask].std(axis=1),.001)
    scaled=np.zeros((EXTRA,*SHAPE),np.float32)
    scaled[:5,coverage]=((summaries[:,coverage]-average[:,None])/
                         spread[:,None]).clip(-5,5)
    hidden=raw[5:,fit_mask].T
    pca=PCA(n_components=8,svd_solver="randomized",random_state=97)
    pca.fit(hidden)
    projected=pca.transform(raw[5:,coverage].T).T
    mean=projected[:,np.flatnonzero(fit_mask[coverage])].mean(axis=1)
    std=np.maximum(projected[:,np.flatnonzero(fit_mask[coverage])].std(axis=1),.05)
    scaled[5:13,coverage]=((projected-mean[:,None])/std[:,None]).clip(-5,5)
    scaled[13]=coverage.astype(np.float32)
    audit={"source_physics_cells":int(fit_mask.sum()),
           "summary_mean":average.round(5).tolist(),
           "summary_std":spread.round(5).tolist(),
           "hidden_pca_variance_ratio":pca.explained_variance_ratio_.round(5).tolist()}
    if return_transform:
        transform={"summary_mean":average,"summary_std":spread,
                   "pca_mean":pca.mean_.astype(np.float32),
                   "pca_components":pca.components_.astype(np.float32),
                   "projected_mean":mean,"projected_std":std}
        return scaled,audit,transform
    return scaled,audit


def apply_physics_transform(raw:np.ndarray,coverage:np.ndarray,
                            transform:dict):
    """Apply the original source-only transform after 10 m fine-tuning."""
    result=np.zeros((EXTRA,*SHAPE),np.float32)
    selected=raw[:,coverage]
    result[:5,coverage]=((selected[:5]-transform["summary_mean"][:,None])/
                         transform["summary_std"][:,None]).clip(-5,5)
    hidden=(selected[5:].T-transform["pca_mean"])@transform["pca_components"].T
    result[5:13,coverage]=((hidden.T-transform["projected_mean"][:,None])/
                           transform["projected_std"][:,None]).clip(-5,5)
    result[13]=coverage.astype(np.float32)
    return result


def make_context(physics:np.ndarray):
    def with_physics(data,day,static):
        image,base,simple=context(data,day,static)
        # Preserve the four raw histories and five Gaussian histories at
        # their exact mask-tested indices 0:4 and 15:20. Validity stays last.
        augmented=np.concatenate((image[:-1],physics,image[-1:]))
        return augmented,base,simple
    return with_physics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold",type=int,choices=range(1,6))
    parser.add_argument("--seed-offset",type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(4)
    data=load_grid()
    with np.load(FEATURES_NPZ) as saved:
        raw=saved["features"]
        coverage=saved["coverage"]
        if not np.array_equal(saved["candidate_flat"],data["candidate_flat"]):
            raise ValueError("Frozen physics features and report grid differ")
    if raw.shape!=(53,*SHAPE) or int(coverage.sum())!=9808 or not np.isfinite(raw).all():
        raise ValueError("Frozen physics feature audit changed")
    z,_,_,terrain_audit=terrain(data)
    district=data["district_id"].ravel()[data["candidate_flat"]]
    covered=coverage.ravel()[data["candidate_flat"]]
    folds=np.array_split(np.random.default_rng(917).permutation(50),5)
    records,states=[],{}
    for fold_id,heldout in enumerate(folds,1):
        if args.fold is not None and fold_id!=args.fold:
            continue
        source=(~np.isin(district,heldout))&covered
        target=np.isin(district,heldout)&covered
        source_grid=np.zeros(SHAPE,bool)
        source_grid.ravel()[data["candidate_flat"]]=source
        static,norm=lidar_static(data,source,z)
        mask_self_check(data,static)
        physics,physics_audit=physics_for_fold(raw,coverage,source_grid)
        record={"fold":fold_id,
                "heldout_district_codes":[data["districts"][j]["code"] for j in heldout],
                "source_cells":int(source.sum()),"heldout_cells":int(target.sum()),
                "static_normalization":norm,"physics_normalization":physics_audit,
                "arms":{}}
        states[fold_id]={}
        print("fold",fold_id,"source",int(source.sum()),"test",int(target.sum()),flush=True)
        for name in ARMS:
            active=physics.copy()
            if name=="history_cnn":
                active[:13]=0
            elif name=="depth_and_hidden_no_vectors":
                active[2:5]=0
            context_fn=make_context(active)
            seed=SEED+fold_id*1000+args.seed_offset
            model,fit=train(data,source,static,seed,ARM,
                            model_factory=FrozenPhysicsReportCNN,
                            context_fn=context_fn)
            valid=score_days(data,source,static,model,
                             range(VALID_START,TEST_START),context_fn=context_fn)
            alpha=max(ALPHAS,key=lambda a:(
                valid[str(a)]["isolated_group_capture"],
                valid[str(a)]["isolated_cell_capture"],-a))
            held=score_days(data,target,static,model,
                            range(TEST_START,N_DAYS),context_fn=context_fn)
            record["arms"][name]={"fit":fit,"source_validation":valid,
                                  "selected_alpha":alpha,"heldout_q4":held}
            states[fold_id][name]={k:v.cpu() for k,v in model.state_dict().items()}
            print("fold",fold_id,name,"alpha",alpha,"Q4 cells",
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
    baseline=pooled[ARMS[0]]["0.0"]
    for arm in ARMS[1:]:
        got=pooled[arm]["0.0"]
        if (got["isolated_positive"]!=baseline["isolated_positive"] or
            got["isolated_groups"]!=baseline["isolated_groups"] or
            abs(got["isolated_hits"]-baseline["isolated_hits"])>1e-4):
            raise ValueError("Matched history queue differs across arms")
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question":"Do frozen 10 m depth, direction and latent maps help a spatial 250 m masked report CNN?",
            "teacher_features":str(FEATURES_NPZ),
            "teacher_audit":str(FEATURES_JSON),
            "report_target":"Traffy report-positive 250 m cell-days, not verified flood extent",
            "architecture":"The 250 m CNN sees all physics layers spatially through five depthwise residual 3x3 steps; learned radius reaches seven 250 m cells. Signed east/north velocity can influence neighboring target cells.",
            "frozen_teacher":True,
            "physics_summary":"Mean and p90 simulated depth, mean signed u/v, mean speed, eight source-only PCA channels of the frozen 10 m trunk, coverage",
            "direction_ablation":"Zeros explicit signed u/v and speed but retains pretrained latent features, which can still contain direction information",
            "training":"2023-24 source districts only; target-centered 3x3 report history and its Gaussian contributions removed; group ranking loss; same width and initialization for all arms",
            "selection":"Jan-Sep 2025 source-district isolated group capture, then cell capture; Q4 excluded districts never select alpha",
            "queue":"10% per day among physics-covered heldout cells, with 20% slots reserved for strict isolated candidates",
            "seed_offset":args.seed_offset,
            "coverage_cells":int(coverage.sum()),
            "terrain_audit":terrain_audit,
            "folds":records,"pooled_heldout_q4":pooled,
            "limitations":["Only 9,808 of 25,812 candidate cells have qualifying terrain source; queue comparisons are on this subset",
                           "Pretraining already saw Bang Kapi/Bang Khae simulated terrain; temporal report labels are held out, but those physical districts are not wholly unseen",
                           "Synthetic fixed rain and closed tile, no observed water state, drains, pumps or infiltration",
                           "Unreported comparisons are unknown, not verified dry",
                           "Q4 2025 report holdout has been inspected repeatedly in this project"]}
    suffix=(f"_seed_{args.seed_offset}" if args.seed_offset else "")
    if args.fold is not None:
        suffix+=f"_fold_{args.fold}"
    out_json=OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    out_model=OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    out_json.write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"states":states},out_model)
    print("pooled",{arm:(rows["source_selected"]["isolated_hits"],
                           rows["source_selected"]["group_hits"])
                    for arm,rows in pooled.items()},flush=True)
    print("saved",out_json,out_model,flush=True)


if __name__=="__main__":
    main()
