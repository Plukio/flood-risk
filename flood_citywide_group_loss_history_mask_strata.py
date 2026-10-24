"""Audit isolated versus adjacent-history sites on the matched Q4 queue."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import label

from flood_citywide_history_ranker import N_DAYS,SHAPE,TEST_START,load_grid
from flood_citywide_isolated_site_ranker import isolated
from flood_citywide_spatial_novel_queue import (
    SpatialNovelCNN,day_maps,queue_weights,static_image,
)
from flood_citywide_spatial_mask_augmentation import (
    OUT_MODEL as PRIOR_MODEL,OUT_JSON as PRIOR_JSON,
)
from flood_citywide_group_loss_history_mask import OUT_MODEL,OUT_JSON
from flood_multi_target_citywide_data import OUT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed-offset",type=int,default=0)
    args=parser.parse_args()
    suffix="" if args.seed_offset==0 else f"_seed_{args.seed_offset}"
    group_model=OUT_MODEL if not suffix else OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    group_json=OUT_JSON if not suffix else OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    dest=OUT / f"q4_2025_group_loss_history_mask_strata{suffix}.json"
    torch.set_num_threads(4)
    data=load_grid()
    group_fit=json.loads(group_json.read_text())
    old_fit=json.loads(PRIOR_JSON.read_text())
    group_states=torch.load(group_model,weights_only=True)["states"]
    old_states=torch.load(PRIOR_MODEL,weights_only=True)["states"]
    district=data["district_id"].ravel()[data["candidate_flat"]]
    splits=np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])),5)
    totals={name:{"isolated_positive":0,"isolated_hits":0.,
                  "adjacent_positive":0,"adjacent_hits":0.,
                  "isolated_groups":0,"isolated_group_hits":0.,
                  "adjacent_groups":0,"adjacent_group_hits":0.}
            for name in ("previous_masked_cnn","group_loss_masked_cnn")}
    for fold_id,ids in enumerate(splits,1):
        target=np.isin(district,ids)
        source=~target
        static,_=static_image(data,source)
        models={}
        for name,state in (
            ("previous_masked_cnn",old_states[fold_id]["synthetic_center_block"]),
            ("group_loss_masked_cnn",group_states[fold_id])):
            model=SpatialNovelCNN().eval()
            model.load_state_dict(state)
            models[name]=model
        for day in range(TEST_START,N_DAYS):
            image,base,_,novel=day_maps(data,day,static)
            truth=data["reports"][day].ravel()[data["candidate_flat"]]>0
            iso=isolated(data,day)[data["candidate"]]
            group_iso=truth[target]&novel[target]&iso[target]
            group_adj=truth[target]&novel[target]&~iso[target]
            scoped_flat=data["candidate_flat"][target]
            memberships={}
            for kind,flags in (("isolated",group_iso),("adjacent",group_adj)):
                grid=np.zeros(SHAPE,bool)
                grid.ravel()[scoped_flat]=flags
                components,count=label(grid,structure=np.ones((3,3),bool))
                ids=components.ravel()[scoped_flat]
                memberships[kind]=[ids==j for j in range(1,count+1)]
            for name,model in models.items():
                scores=model.full_scores(image,data["candidate"])
                w=queue_weights(base[target],scores[target],novel[target],.20)
                row=totals[name]
                row["isolated_positive"]+=int(group_iso.sum())
                row["isolated_hits"]+=float(w[group_iso].sum())
                row["adjacent_positive"]+=int(group_adj.sum())
                row["adjacent_hits"]+=float(w[group_adj].sum())
                for kind,members in memberships.items():
                    row[kind+"_groups"]+=len(members)
                    row[kind+"_group_hits"]+=sum(float(w[m].max())
                                                   for m in members)
    for row in totals.values():
        if row["isolated_positive"]!=332 or row["adjacent_positive"]!=1332:
            raise ValueError("Matched subgroup denominators changed")
        row["isolated_capture"]=round(
            row["isolated_hits"]/row["isolated_positive"],5)
        row["adjacent_capture"]=round(
            row["adjacent_hits"]/row["adjacent_positive"],5)
        for kind in ("isolated","adjacent"):
            row[kind+"_group_capture"]=round(
                row[kind+"_group_hits"]/row[kind+"_groups"],5)
        row["first_in_year_capture"]=round(
            (row["isolated_hits"]+row["adjacent_hits"])/1664,5)
    expected=old_fit["pooled_fold_queues"]["synthetic_center_block"]["novel_capture"]
    measured=totals["previous_masked_cnn"]["first_in_year_capture"]
    if abs(measured-expected)>1e-5 or abs(
        totals["group_loss_masked_cnn"]["first_in_year_capture"]-
        group_fit["pooled_heldout_q4"]["novel_capture"])>1e-5:
        raise ValueError("Subgroup audit does not reconstruct frozen result")
    payload={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "seed_offset":args.seed_offset,
             "queue":"Same five Q4 district folds and 20% first-in-year slot reserve",
             "definition":"Isolated means no own or eight adjacent report in prior 365 days",
             "methods":totals}
    dest.write_text(json.dumps(payload,indent=2)+"\n")
    print(totals,"saved",dest,flush=True)


if __name__=="__main__":
    main()
