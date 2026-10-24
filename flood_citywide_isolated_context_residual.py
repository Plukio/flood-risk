"""Test whether multiscale spatial context improves isolated-site ranking.

The CNN receives 15x15 physical/history patches plus pre-day report smooths at
750 m, 1.5 km and 3 km scales. It learns a regularized correction to the
frozen distant-history score. Correction strength is selected on source-only
2025 validation, then evaluated on Q4 outcomes in excluded districts.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import label
from torch import nn
from torch.nn import functional as F

from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_isolated_site_ranker import (
    day_inputs, isolated, queue, samples,
)
from flood_citywide_spatial_novel_queue import (
    STATIC_NAMES, SpatialNovelCNN, static_image,
)
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_isolated_context_residual_fivefold.json"
OUT_MODEL = OUT / "q4_2025_isolated_context_residual_fivefold.pt"
ALPHAS = (0., .1, .25, .5, 1.)
EPOCHS = 2


class ContextResidualCNN(SpatialNovelCNN):
    """Seven-cell learned radius, with wider history context as input layers."""

    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(4 + len(STATIC_NAMES) + 2 + 5, 8, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


def context(data: dict, day: int, static: np.ndarray):
    image, base, simple, _, distant = day_inputs(
        data, day, static, include_distant_layers=True)
    # SpatialNovelCNN treats the last channel as the grid-validity mask.
    return np.concatenate((image[:-1], distant, image[-1:])).astype(np.float32), base, simple


def train(data: dict, source: np.ndarray, static: np.ndarray, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    source_grid = np.zeros(SHAPE, bool)
    source_grid.ravel()[data["candidate_flat"]] = source
    days = [d for d in range(TRAIN_START, VALID_START)
            if np.any((data["reports"][d] > 0) &
                      isolated(data, d) & source_grid)]
    model = ContextResidualCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    losses, groups = [], 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for day in rng.permutation(days):
            image, _, simple = context(data, int(day), static)
            picked = samples(data, int(day), source_grid, simple, rng, limit=12)
            if picked is None:
                continue
            pos, neg, n_groups = picked
            groups += n_groups
            indices = np.r_[pos, neg]
            batch = patches(image, data["candidate_flat"], indices)
            if torch.any(batch[:, :4, RADIUS-1:RADIUS+2,
                                      RADIUS-1:RADIUS+2] != 0):
                raise ValueError("Isolated candidate has immediate report history")
            residual = model.patch_scores(batch)
            fixed = torch.from_numpy(np.ascontiguousarray(simple[indices]))
            score = fixed + residual
            loss = F.softplus(score[len(pos):]-score[:len(pos)]).mean()
            loss = loss + .001*residual.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(round(float(np.mean(epoch_losses)),5))
        print("epoch",epoch+1,"loss",losses[-1],flush=True)
    model.eval()
    return model,{"train_days_with_isolated_positive":len(days),
                  "positive_groups_sampled_across_epochs":groups,
                  "epoch_pairwise_loss":losses}


def score_days(data: dict, scope: np.ndarray, static: np.ndarray,
               model: ContextResidualCNN, days: range, context_fn=context):
    flat = data["candidate_flat"]
    chosen = np.flatnonzero(scope)
    chosen_flat = flat[chosen]
    names = [str(a) for a in ALPHAS]
    rows = {name:{"all_positive":0,"all_hits":0.,
                  "isolated_positive":0,"isolated_hits":0.,
                  "isolated_groups":0,"group_hits":0.,
                  "daily_isolated_positive":[],"daily_isolated_hits":[]}
            for name in names}
    for day in days:
        image, base, simple = context_fn(data, day, static)
        residual = model.full_scores(image, data["candidate"])
        eligible = isolated(data, day)[data["candidate"]][chosen]
        truth = (data["reports"][day].ravel()[flat] > 0)[chosen]
        iso_y = truth & eligible
        grid = np.zeros(SHAPE, bool)
        grid.ravel()[chosen_flat] = iso_y
        components, n_groups = label(grid, structure=np.ones((3,3),bool))
        ids = components.ravel()[chosen_flat]
        for alpha in ALPHAS:
            w = queue(base[chosen],
                      simple[chosen]+alpha*residual[chosen], eligible)
            row = rows[str(alpha)]
            row["all_positive"] += int(truth.sum())
            row["all_hits"] += float(w[truth].sum())
            row["isolated_positive"] += int(iso_y.sum())
            hits = float(w[iso_y].sum())
            row["isolated_hits"] += hits
            row["isolated_groups"] += n_groups
            row["group_hits"] += sum(float(w[ids==j].max())
                                     for j in range(1,n_groups+1))
            row["daily_isolated_positive"].append(int(iso_y.sum()))
            row["daily_isolated_hits"].append(round(hits,5))
    for row in rows.values():
        row["all_capture"] = round(row["all_hits"]/row["all_positive"],5)
        row["isolated_cell_capture"] = round(
            row["isolated_hits"]/row["isolated_positive"],5)
        row["isolated_group_capture"] = round(
            row["group_hits"]/row["isolated_groups"],5)
    return rows


def pool(rows: list[dict]):
    keys = ("all_positive","all_hits","isolated_positive","isolated_hits",
            "isolated_groups","group_hits")
    out = {key:sum(r[key] for r in rows) for key in keys}
    out["all_capture"] = round(out["all_hits"]/out["all_positive"],5)
    out["isolated_cell_capture"] = round(
        out["isolated_hits"]/out["isolated_positive"],5)
    out["isolated_group_capture"] = round(
        out["group_hits"]/out["isolated_groups"],5)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed-offset",type=int,default=0)
    args=parser.parse_args()
    suffix="" if args.seed_offset==0 else f"_seed_{args.seed_offset}"
    out_json=OUT_JSON if not suffix else OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    out_model=OUT_MODEL if not suffix else OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data=load_grid()
    district=data["district_id"].ravel()[data["candidate_flat"]]
    district_folds=np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])),5)
    folds, states=[],{}
    for fold_id, ids in enumerate(district_folds,1):
        source=~np.isin(district,ids)
        print("fold",fold_id,flush=True)
        static,norm=static_image(data,source)
        model,fit=train(data,source,static,SEED+fold_id*1000+args.seed_offset)
        valid=score_days(data,source,static,model,range(VALID_START,TEST_START))
        # Source-only selection. On equal group scores, prefer the smaller
        # correction, leaving the fixed baseline selected for a full tie.
        chosen=max(ALPHAS,key=lambda a:(valid[str(a)]["isolated_group_capture"],
                                       valid[str(a)]["isolated_cell_capture"],-a))
        test=score_days(data,~source,static,model,range(TEST_START,N_DAYS))
        folds.append({"fold":fold_id,
                      "heldout_district_codes":[data["districts"][j]["code"]
                                                for j in ids],
                      "normalization":norm,"fit":fit,
                      "source_validation":valid,"selected_alpha":chosen,
                      "heldout_q4":test})
        states[fold_id]={k:v.cpu() for k,v in model.state_dict().items()}
        print("fold",fold_id,"alpha",chosen,"q4",
              {name:(row["isolated_cell_capture"],
                     row["isolated_group_capture"])
               for name,row in test.items()},flush=True)
    pooled={str(a):pool([f["heldout_q4"][str(a)] for f in folds]) for a in ALPHAS}
    pooled["source_selected"]=pool([
        f["heldout_q4"][str(f["selected_alpha"])] for f in folds])
    if (pooled["0.0"]["all_positive"]!=7209 or
        pooled["0.0"]["isolated_positive"]!=332 or
        pooled["0.0"]["isolated_groups"]!=158 or
        pooled["0.0"]["isolated_hits"]!=65.):
        raise ValueError("Frozen isolated queue changed")
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed_offset":args.seed_offset,
            "question":"Can multiscale spatial CNN context improve on distant report history for isolated sites?",
            "target":"Traffy report-positive 250m cell-days, not verified flooding",
            "dates":{"training":["2023-01-01","2024-12-31"],
                     "source_validation":["2025-01-01","2025-09-30"],
                     "heldout_q4":["2025-10-01","2025-12-31"]},
            "features":"Pre-day report maps and 750m/1500m/3000m smooths; terrain, OSM roads/canals, WorldCover; no future rain",
            "model":"SpatialNovelCNN steps with five extra distant-history layers; zero-initialized residual head over fixed distant-history score",
            "loss":"paired isolated-group representative versus unreported isolated comparison; regularized residual",
            "queue":"10% of heldout fold cells per day, 20% of slots reserved for isolated candidates",
            "alpha_candidates":ALPHAS,
            "alpha_selection":"highest source-validation isolated group capture; then cell capture; then smaller alpha",
            "folds":folds,"pooled_heldout_q4":pooled,
            "limitations":["Unreported comparisons are unknown, not verified dry",
                           "Touching positive groups are not original complaint IDs",
                           "Q4 2025 has been inspected repeatedly",
                           "Combined district and time holdout does not isolate each shift",
                           "No verified water extent or rain-scenario validity"]}
    out_json.write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"states":states},out_model)
    print("pooled",{k:(v["isolated_cell_capture"],
                        v["isolated_group_capture"],v["all_capture"])
                    for k,v in pooled.items()},flush=True)


if __name__=="__main__":
    main()
