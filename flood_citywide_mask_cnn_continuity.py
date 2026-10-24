"""Does terrain-gated score continuity improve the masked CNN report queue?"""
from __future__ import annotations

import json
from datetime import datetime,timezone

import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS,TEST_START,VALID_START,load_grid
from flood_citywide_spatial_mask_augmentation import OUT_MODEL
from flood_citywide_spatial_novel_queue import SpatialNovelCNN,collect,static_image
from flood_citywide_target_site_mask_test import q20_rows
from flood_citywide_terrain_continuity import links
from flood_multi_target_citywide_data import OUT

OUTPUT=OUT/"q4_2025_mask_cnn_terrain_continuity_fivefold.json"
ALPHAS=(0.,.15,.3,.5,1.)


def spatial_message(score,novel,graph):
    eligible=novel.astype(np.float32)
    numerator=graph.dot((score*eligible).T).T
    denominator=graph.dot(eligible.T).T
    return np.divide(numerator,denominator,out=score.copy(),where=denominator>1e-8)


def metrics(row):
    return {"positive":row["positive_cell_days"],
            "novel_positive":row["novel_positive_cell_days"],
            "hit":round(sum(row["daily_captured"]),5),
            "novel_hit":round(sum(row["daily_novel_captured"]),5),
            "overall_capture":row["top_10pct_capture"],
            "novel_capture":row["novel_capture"],
            "daily_positive":row["daily_positive"],
            "daily_captured":row["daily_captured"],
            "daily_novel_positive":row["daily_novel_positive"],
            "daily_novel_captured":row["daily_novel_captured"]}


def evaluate_arms(data,arrays,scope,graphs):
    base,simple,score,novel,truth=arrays
    result={}
    for name,graph in graphs.items():
        message=spatial_message(score,novel,graph)
        for alpha in ALPHAS:
            adjusted=(1-alpha)*score+alpha*message
            row=q20_rows(data,(base,simple,adjusted,novel,truth),scope,False)["model"]
            result[f"{name}_{alpha:g}"]=metrics(row)
    return result


def main():
    torch.set_num_threads(4)
    data=load_grid()
    ordinary,physical,a,b,dz,audit=links(data)
    graphs={"ordinary":ordinary,"terrain":physical}
    states=torch.load(OUT_MODEL,weights_only=True)["states"]
    district=data["district_id"].ravel()[data["candidate_flat"]]
    splits=np.array_split(np.random.default_rng(917).permutation(len(data["districts"])),5)
    folds=[]
    for number,ids in enumerate(splits,1):
        target=np.isin(district,ids)
        source=~target
        static,_=static_image(data,source)
        model=SpatialNovelCNN().eval()
        model.load_state_dict(states[number]["synthetic_center_block"])
        print("fold",number,"source scores",flush=True)
        val=evaluate_arms(data,collect(data,model,static,range(VALID_START,TEST_START)),source,graphs)
        selected={name:max(ALPHAS,key=lambda x:(val[f"{name}_{x:g}"]["novel_capture"],-x))
                  for name in graphs}
        print("fold",number,"selected",selected,"Q4 scores",flush=True)
        q4=evaluate_arms(data,collect(data,model,static,range(TEST_START,N_DAYS)),target,graphs)
        folds.append({"fold":number,"source_selected_alpha":selected,
                      "source_validation":{k:{"overall_capture":v["overall_capture"],
                                              "novel_capture":v["novel_capture"]}
                                           for k,v in val.items()},"q4":q4})
        print("fold",number,"Q4",{k:v["novel_capture"] for k,v in q4.items()},flush=True)
    def pool(choice):
        rows=[f["q4"][choice(f)] for f in folds]
        positive=sum(r["positive"] for r in rows)
        novel=sum(r["novel_positive"] for r in rows)
        hit=sum(r["hit"] for r in rows)
        novel_hit=sum(r["novel_hit"] for r in rows)
        return {"positive":positive,"novel_positive":novel,
                "hit":round(hit,5),"novel_hit":round(novel_hit,5),
                "overall_capture":round(hit/positive,5),
                "novel_capture":round(novel_hit/novel,5),
                "daily_positive":np.sum([r["daily_positive"] for r in rows],axis=0).astype(int).tolist(),
                "daily_captured":np.sum([r["daily_captured"] for r in rows],axis=0).round(5).tolist(),
                "daily_novel_positive":np.sum([r["daily_novel_positive"] for r in rows],axis=0).astype(int).tolist(),
                "daily_novel_captured":np.sum([r["daily_novel_captured"] for r in rows],axis=0).round(5).tolist()}
    pooled={"original":pool(lambda f:"ordinary_0")}
    for kind in graphs:
        pooled[f"{kind}_source_selected"]=pool(lambda f,kind=kind:
            f"{kind}_{f['source_selected_alpha'][kind]:g}")
        for alpha in ALPHAS[1:]:
            pooled[f"{kind}_{alpha:g}"]=pool(lambda f,kind=kind,alpha=alpha:f"{kind}_{alpha:g}")
    if (pooled["original"]["positive"]!=7209 or
        pooled["original"]["novel_positive"]!=1664 or
        pooled["original"]["novel_capture"]!=.13529):
        raise ValueError("Original queue changed")
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task":"same 10%-of-cells, 20%-novel-reserve daily Q4 report queue",
            "method":"one-hop weighted neighbor average of saved CNN logits over pre-day novel candidates; blend strength selected on source validation",
            "graph_audit":audit,"alphas":ALPHAS,"folds":folds,"pooled_q4":pooled,
            "limitations":["Terrain edge is a hypothesis, not proof of hydraulic connection",
                           "No complete hard-barrier map or event-time gate/pump state",
                           "Traffy report absence does not establish dry cells",
                           "Q4 has been examined repeatedly"]}
    OUTPUT.write_text(json.dumps(result,indent=2)+"\n")
    print("pooled",{k:(v["novel_capture"],v["overall_capture"])
                    for k,v in pooled.items()},flush=True)
    print("saved",OUTPUT,flush=True)


if __name__=="__main__":
    main()
