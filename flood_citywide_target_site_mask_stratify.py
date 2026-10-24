"""Audit novel-site capture by presence of pre-day adjacent report history.

Uses the saved five held-out-district models from the matched masking test.
Adjacent means any of the eight neighboring 250 m cells had a report in the
preceding 365 days; this does not identify a shared original complaint.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import convolve

from flood_citywide_history_ranker import (
    N_DAYS, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_spatial_novel_queue import (
    SpatialNovelCNN, day_maps, queue_weights, static_image,
)
from flood_citywide_target_site_mask_test import (
    ARMS, OUT_JSON, OUT_MODEL, QUOTA,
)
from flood_multi_target_citywide_data import OUT

DEST = OUT / "q4_2025_target_site_mask_adjacent_strata.json"
METHODS = ("simple_neighbor", *ARMS)


def adjacent_positive_audit(data: dict, days: range, kernel: np.ndarray) -> dict:
    """Count positives with pre-day neighbor history, split by own history."""
    candidate=data["candidate"]
    result={"first_in_year":{"positive_cell_days":0,"with_adjacent_history":0},
            "repeat":{"positive_cell_days":0,"with_adjacent_history":0}}
    for day in days:
        prior=(data["prefix"][day]-data["prefix"][day-365]) > 0
        adjacent=(convolve(prior.astype(np.uint8),kernel,
                           mode="constant") > 0)[candidate]
        truth=data["reports"][day,candidate] > 0
        own=prior[candidate]
        for name,mask in (("first_in_year",~own),("repeat",own)):
            row=result[name]
            row["positive_cell_days"]+=int((truth & mask).sum())
            row["with_adjacent_history"]+=int((truth & mask & adjacent).sum())
    for row in result.values():
        row["adjacent_share"]=round(
            row["with_adjacent_history"]/row["positive_cell_days"],5)
    return result


def main() -> None:
    torch.set_num_threads(4)
    data = load_grid()
    result = json.loads(OUT_JSON.read_text())
    states = torch.load(OUT_MODEL, weights_only=True)["states"]
    candidate = data["candidate"]
    flat = data["candidate_flat"]
    kernel = np.ones((3,3), np.uint8)
    kernel[1,1] = 0
    training_audit=adjacent_positive_audit(
        data,range(TRAIN_START,VALID_START),kernel)
    q4_audit=adjacent_positive_audit(data,range(TEST_START,N_DAYS),kernel)
    adjacent = []
    for day in range(TEST_START,N_DAYS):
        prior = (data["prefix"][day]-data["prefix"][day-365]) > 0
        adjacent.append((convolve(prior.astype(np.uint8),kernel,
                                   mode="constant") > 0)[candidate])
    adjacent = np.stack(adjacent)
    district = data["district_id"].ravel()[flat]
    totals = {str(flag):{"positive_cell_days":0,
                         **{name:0. for name in METHODS}}
              for flag in (False,True)}
    per_fold=[]
    for fold in result["folds"]:
        number = fold["fold"]
        codes=set(fold["district_codes"])
        target=np.array([data["districts"][i]["code"] in codes
                         for i in district])
        source=~target
        static,_=static_image(data,source)
        models={}
        for arm in ARMS:
            model=SpatialNovelCNN().eval()
            model.load_state_dict(states[number][arm])
            models[arm]=model
        fold_totals={str(flag):{"positive_cell_days":0,
                                **{name:0. for name in METHODS}}
                     for flag in (False,True)}
        for offset,day in enumerate(range(TEST_START,N_DAYS)):
            image,base,simple,novel=day_maps(data,day,static)
            truth=data["reports"][day].ravel()[flat] > 0
            scoped_novel=novel[target]
            picks={"simple_neighbor":queue_weights(
                base[target],simple[target],scoped_novel,QUOTA)}
            for arm,model in models.items():
                score=model.full_scores(image,candidate)
                picks[arm]=queue_weights(base[target],score[target],
                                        scoped_novel,QUOTA)
            for flag in (False,True):
                group=scoped_novel & truth[target] & (adjacent[offset,target]==flag)
                row=fold_totals[str(flag)]
                row["positive_cell_days"]+=int(group.sum())
                for name,w in picks.items():
                    row[name]+=float(w[group].sum())
        expected=fold["q4"]["simple_neighbor"]["novel_positive_cell_days"]
        if sum(v["positive_cell_days"] for v in fold_totals.values())!=expected:
            raise ValueError(f"Fold {number} novel-positive denominator changed")
        per_fold.append({"fold":number,"strata":fold_totals})
        for flag in (False,True):
            row=totals[str(flag)]
            part=fold_totals[str(flag)]
            for key in row:
                row[key]+=part[key]
        print("fold",number,"strata",fold_totals,flush=True)
    for row in totals.values():
        row["capture"]={name:round(row[name]/row["positive_cell_days"],5)
                        for name in METHODS}
    if sum(v["positive_cell_days"] for v in totals.values())!=1664:
        raise ValueError("Pooled novel-positive denominator changed")
    payload={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "definition":"Any of eight neighboring 250 m cells has a report in prior 365 days",
             "target":"first-in-a-year report-positive cell-days, not verified flooding",
             "period":["2025-10-01","2025-12-31"],
             "adjacent_positive_audit":{"training_2023_2024":training_audit,
                                        "q4_2025":q4_audit},
             "strata":totals,"folds":per_fold}
    DEST.write_text(json.dumps(payload,indent=2)+"\n")
    print("pooled",totals,"saved",DEST,flush=True)


if __name__=="__main__":
    main()
