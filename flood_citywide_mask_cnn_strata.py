"""Check a source-selected isolated-site subqueue within the fixed novel reserve."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from flood_citywide_history_ranker import N_DAYS, TEST_START, VALID_START, load_grid
from flood_citywide_spatial_mask_augmentation import OUT_MODEL as BLOCK_MODEL
from flood_citywide_spatial_novel_queue import (
    SpatialNovelCNN, day_maps, queue_weights, static_image, top_weights,
)
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_mask_cnn_strata_fivefold.json"
CENTER_MODEL = OUT / "q4_2025_target_site_mask_fivefold.pt"
SHARES = (0., .05, .1, .2, .3, .5)  # share of novel reserve for isolated cells


def models_for_fold(block_state, center_state):
    models = []
    for state in (block_state, center_state):
        model = SpatialNovelCNN()
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def score_period(data, source, static, models, days):
    indices = np.flatnonzero(source)
    flat = data["candidate_flat"]
    result = {str(s): {"positive":0,"novel_positive":0,
                        "isolated_positive":0,"captured":0.,
                        "novel_captured":0.,"isolated_captured":0.}
              for s in SHARES}
    for day in days:
        image, base, _, novel = day_maps(data,day,static)
        with torch.no_grad():
            block = models[0].full_scores(image,data["candidate"])
            center = models[1].full_scores(image,data["candidate"])
        isolated = (maximum_filter(image[0],size=3,mode="constant") == 0)[data["candidate"]]
        y = data["reports"][day].ravel()[flat][indices]
        n = novel[indices]
        iso = n & isolated[indices]
        adj = n & ~iso
        budget = .1*len(indices)
        novel_budget = .2*budget
        known = ~n
        for share in SHARES:
            if share == 0:
                w = queue_weights(base[indices],block[indices],n,.2)
            else:
                w = np.zeros(len(indices),np.float32)
                w[known] = top_weights(base[indices][known],budget-novel_budget)
                iso_budget = min(novel_budget*share,int(iso.sum()))
                adj_budget = novel_budget-iso_budget
                if adj_budget > adj.sum():
                    iso_budget += adj_budget-adj.sum()
                    adj_budget = float(adj.sum())
                w[iso] = top_weights(center[indices][iso],iso_budget)
                w[adj] = top_weights(block[indices][adj],adj_budget)
                if abs(float(w.sum())-budget)>.02:
                    raise ValueError("Daily capacity changed")
            row = result[str(share)]
            row["positive"] += int(y.sum())
            row["novel_positive"] += int(y[n].sum())
            row["isolated_positive"] += int(y[iso].sum())
            row["captured"] += float(np.dot(w,y))
            row["novel_captured"] += float(np.dot(w,n*y))
            row["isolated_captured"] += float(np.dot(w,iso*y))
    for row in result.values():
        row["overall_capture"] = round(row["captured"]/row["positive"],5)
        row["novel_capture"] = round(row["novel_captured"]/row["novel_positive"],5)
        row["isolated_capture"] = round(row["isolated_captured"]/row["isolated_positive"],5)
        for key in ("captured","novel_captured","isolated_captured"):
            row[key] = round(row[key],5)
    return result


def main():
    torch.set_num_threads(4)
    data = load_grid()
    block_states = torch.load(BLOCK_MODEL,weights_only=True)["states"]
    center_states = torch.load(CENTER_MODEL,weights_only=True)["states"]
    district = data["district_id"].ravel()[data["candidate_flat"]]
    district_folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])),5)
    folds = []
    for fold_number,ids in enumerate(district_folds,1):
        target = np.isin(district,ids)
        source = ~target
        static,_ = static_image(data,source)
        models = models_for_fold(block_states[fold_number]["synthetic_center_block"],
                                 center_states[fold_number]["target_centered_3x3"])
        print("fold",fold_number,"source validation",flush=True)
        validation = score_period(data,source,static,models,
                                  range(VALID_START,TEST_START))
        chosen = max(SHARES,key=lambda s:(validation[str(s)]["novel_capture"],-s))
        print("fold",fold_number,"selected",chosen,"Q4",flush=True)
        q4 = score_period(data,target,static,models,range(TEST_START,N_DAYS))
        folds.append({"fold":fold_number,"source_validation":validation,
                      "source_selected_share":chosen,"q4":q4})
        print("fold",fold_number,"Q4 novel",{s:q4[str(s)]["novel_capture"]
                                            for s in SHARES},flush=True)
    def pool(choice):
        rows = [f["q4"][str(choice(f))] for f in folds]
        total = {k:round(sum(r[k] for r in rows),5) for k in
                 ("positive","novel_positive","isolated_positive",
                  "captured","novel_captured","isolated_captured")}
        total.update({"overall_capture":round(total["captured"]/total["positive"],5),
                      "novel_capture":round(total["novel_captured"]/total["novel_positive"],5),
                      "isolated_capture":round(total["isolated_captured"]/total["isolated_positive"],5)})
        return total
    pooled = {str(s):pool(lambda f,s=s:s) for s in SHARES}
    pooled["source_selected"] = pool(lambda f:f["source_selected_share"])
    if (pooled["0.0"]["novel_positive"]!=1664 or
        pooled["0.0"]["novel_capture"]!=.13529):
        raise ValueError("Baseline queue did not reproduce")
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task":"same 20% novel reserve and 10% total daily queue",
            "method":"split novel reserve between adjacent-history cells scored by original block-masked CNN and isolated cells scored by target-centered CNN",
            "selection":"January-September 2025 source district novel capture; Q4 target districts held out from share selection",
            "shares":SHARES,"folds":folds,"pooled_q4":pooled,
            "caution":"both CNN mask arms were previously inspected on Q4; exploratory"}
    OUT_JSON.write_text(json.dumps(result,indent=2)+"\n")
    print("pooled",{k:(v["novel_capture"],v["isolated_capture"])
                     for k,v in pooled.items()},flush=True)
    print("saved",OUT_JSON,flush=True)


if __name__ == "__main__":
    main()
