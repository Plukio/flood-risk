"""Build an interactive Q4 Bangkok map for the masked CNN and history queue.

The exported rank bounds reproduce fractional top-k ties in the original
five-fold backtest. No Q4 outcome enters either model score.
"""
from __future__ import annotations

import base64
import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, load_grid
from flood_citywide_spatial_mask_augmentation import OUT_JSON as EVAL_JSON, OUT_MODEL
from flood_citywide_mask_cnn_continuity import OUTPUT as CONTINUITY_JSON
from flood_citywide_spatial_novel_queue import SpatialNovelCNN, day_maps, static_image
from flood_citywide_terrain_continuity import links
from flood_multi_target_citywide_data import OUT, X0, Y0, ROWS, TILE_M
from flood_multi_target_data import START

OUTPUT = OUT / "q4_2025_mask_cnn_history_interactive_map.html"
TEMPLATE = Path(__file__).with_name("flood_citywide_q4_queue_map_template.html")
DTYPE = np.dtype([("baseline_low", "<u2"), ("baseline_high", "<u2"),
                  ("cnn_low", "<u2"), ("cnn_high", "<u2"),
                  ("smooth_low", "<u2"), ("smooth_high", "<u2"),
                  ("novel", "u1"), ("report", "u1"),
                  ("terrain_candidate", "u1")])


def bounds(values: np.ndarray):
    """Number strictly above and number at/above each score, including ties."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    above = len(values)-cumulative[inverse]
    through_tie = above+counts[inverse]
    if through_tie.max(initial=0)>np.iinfo(np.uint16).max:
        raise ValueError("Rank exceeds packed data type")
    return above.astype(np.uint16), through_tie.astype(np.uint16)


def fractional(low, high, capacity):
    return np.clip((capacity-low)/(high-low),0,1)


def district_paths(districts):
    north = Y0+ROWS*TILE_M
    result=[]
    for d in districts:
        paths=[]
        poly=d["polygon"].simplify(70,preserve_topology=True)
        for part in ([poly] if poly.geom_type=="Polygon" else poly.geoms):
            xy=np.asarray(part.exterior.coords)
            paths.append(np.round(np.column_stack(((xy[:,0]-X0)/250,
                                                  (north-xy[:,1])/250)),2).tolist())
        point=d["polygon"].representative_point()
        result.append({"code":d["code"],"name":d["name_en"],
                       "center":[round((point.x-X0)/250,2),
                                 round((north-point.y)/250,2)],
                       "paths":paths})
    return result


def main():
    torch.set_num_threads(4)
    data=load_grid()
    reference=json.loads(EVAL_JSON.read_text())["pooled_fold_queues"]
    continuity=json.loads(CONTINUITY_JSON.read_text())
    _,terrain_graph,edge_a,edge_b,edge_gap,edge_audit=links(data)
    display_edge=edge_gap<=edge_audit["display_max_gap_m"]
    halo_a,halo_b=edge_a[display_edge],edge_b[display_edge]
    states=torch.load(OUT_MODEL,weights_only=True)["states"]
    flat=data["candidate_flat"]
    n=len(flat)
    district_id=data["district_id"].ravel()[flat]
    fold_id=np.full(n,255,np.uint8)
    for f,ids in enumerate(np.array_split(np.random.default_rng(917).permutation(
            len(data["districts"])),5)):
        fold_id[np.isin(district_id,ids)]=f
    if np.any(fold_id==255):
        raise ValueError("Candidate without evaluation fold")
    fold_indices=[np.flatnonzero(fold_id==f) for f in range(5)]
    statics=[]
    models=[]
    for f in range(5):
        source=fold_id!=f
        static,_=static_image(data,source)
        model=SpatialNovelCNN().eval()
        model.load_state_dict(states[f+1]["synthetic_center_block"])
        statics.append(static)
        models.append(model)
    packed=np.empty((N_DAYS-TEST_START,n),dtype=DTYPE)
    total={name:{"positive":0,"novel_positive":0,"hit":0.,"novel_hit":0.}
           for name in ("baseline","cnn","continuity")}
    fold_counts=np.bincount(fold_id,minlength=5).tolist()
    for offset,day in enumerate(range(TEST_START,N_DAYS)):
        image,base,simple,novel=day_maps(data,day,statics[0])
        report=data["reports"][day].ravel()[flat]
        if report.max(initial=0)>1:
            raise ValueError("Map format expects binary report-positive cells")
        packed[offset]["novel"]=novel.astype(np.uint8)
        packed[offset]["report"]=report
        near=np.zeros(n,np.uint8)
        near[halo_b[report[halo_a]>0]]=1
        near[halo_a[report[halo_b]>0]]=1
        near[report>0]=0
        packed[offset]["terrain_candidate"]=near
        for f,idx in enumerate(fold_indices):
            if f:
                image,_,_,_=day_maps(data,day,statics[f])
            with torch.no_grad():
                predicted=models[f].full_scores(image,data["candidate"])
            eligible=novel.astype(np.float32)
            numer=terrain_graph.dot(predicted*eligible)
            denom=terrain_graph.dot(eligible)
            message=np.divide(numer,denom,out=predicted.copy(),where=denom>1e-8)
            alpha=continuity["folds"][f]["source_selected_alpha"]["terrain"]
            smoothed=(1-alpha)*predicted+alpha*message
            local_novel=novel[idx]
            known_idx=idx[~local_novel]
            novel_idx=idx[local_novel]
            kl,kh=bounds(base[known_idx])
            sl,sh=bounds(simple[novel_idx])
            cl,ch=bounds(predicted[novel_idx])
            ml,mh=bounds(smoothed[novel_idx])
            for group_idx,lo,hi in ((known_idx,kl,kh),(novel_idx,sl,sh)):
                packed[offset]["baseline_low"][group_idx]=lo
                packed[offset]["baseline_high"][group_idx]=hi
            packed[offset]["cnn_low"][known_idx]=kl
            packed[offset]["cnn_high"][known_idx]=kh
            packed[offset]["cnn_low"][novel_idx]=cl
            packed[offset]["cnn_high"][novel_idx]=ch
            packed[offset]["smooth_low"][known_idx]=kl
            packed[offset]["smooth_high"][known_idx]=kh
            packed[offset]["smooth_low"][novel_idx]=ml
            packed[offset]["smooth_high"][novel_idx]=mh
            # Check the packed selection rule at the published 10% area.
            budget=.1*len(idx)
            novel_budget=.2*budget
            known_budget=budget-novel_budget
            if known_budget>len(known_idx) or novel_budget>len(novel_idx):
                raise ValueError("Unexpected budget reallocation at 10%")
            for name,low_name,high_name in (("baseline","baseline_low","baseline_high"),
                                            ("cnn","cnn_low","cnn_high"),
                                            ("continuity","smooth_low","smooth_high")):
                w=np.zeros(len(idx),np.float32)
                for mask,cap in ((~local_novel,known_budget),(local_novel,novel_budget)):
                    selected=idx[mask]
                    w[mask]=fractional(packed[offset][low_name][selected],
                                       packed[offset][high_name][selected],cap)
                row=total[name]
                row["positive"]+=int(report[idx].sum())
                row["novel_positive"]+=int(report[novel_idx].sum())
                row["hit"]+=float(np.dot(w,report[idx]))
                row["novel_hit"]+=float(np.dot(w[local_novel],report[novel_idx]))
        if offset%15==0 or offset==len(packed)-1:
            print("exported",offset+1,"/",len(packed),flush=True)
    for name,expected in (("baseline",reference["simple_neighbor"]),
                          ("cnn",reference["synthetic_center_block"]),
                          ("continuity",continuity["pooled_q4"]["terrain_source_selected"])):
        row=total[name]
        overall=row["hit"]/row["positive"]
        novel=row["novel_hit"]/row["novel_positive"]
        expected_overall=expected.get("top_10pct_capture",expected.get("overall_capture"))
        if (row["positive"]!=7209 or row["novel_positive"]!=1664 or
            abs(overall-expected_overall)>2e-5 or
            abs(novel-expected["novel_capture"])>2e-5):
            raise ValueError(f"Packed map does not reproduce {name}: {overall}, {novel}")
        print(name,"overall",round(overall,5),"novel",round(novel,5),flush=True)
    rr,cc=np.unravel_index(flat,SHAPE)
    stat=data["static"]
    meta={"title":"Bangkok Q4 2025 report queue",
          "width":SHAPE[1],"height":SHAPE[0],"n_cells":n,
          "n_days":len(packed),"row":rr.tolist(),"col":cc.tolist(),
          "fold":fold_id.tolist(),"fold_cells":fold_counts,
          "district_id":district_id.tolist(),
          "built":np.clip(np.round(stat[:,7]*255),0,255).astype(np.uint8).tolist(),
          "canal":(stat[:,3]>0).astype(np.uint8).tolist(),
          "road":(stat[:,5]>0).astype(np.uint8).tolist(),
          "districts":district_paths(data["districts"]),
          "dates":[str(START+timedelta(days=d)) for d in range(TEST_START,N_DAYS)],
          "record_bytes":DTYPE.itemsize,
          "continuity_display_gap_m":edge_audit["display_max_gap_m"],
          "continuity_alphas":[f["source_selected_alpha"]["terrain"]
                               for f in continuity["folds"]],
          "reference":{"baseline_overall":reference["simple_neighbor"]["top_10pct_capture"],
                       "cnn_overall":reference["synthetic_center_block"]["top_10pct_capture"],
                       "baseline_novel":reference["simple_neighbor"]["novel_capture"],
                       "cnn_novel":reference["synthetic_center_block"]["novel_capture"],
                       "continuity_overall":continuity["pooled_q4"]["terrain_source_selected"]["overall_capture"],
                       "continuity_novel":continuity["pooled_q4"]["terrain_source_selected"]["novel_capture"]}}
    encoded=base64.b64encode(gzip.compress(packed.tobytes(),compresslevel=9)).decode("ascii")
    template=TEMPLATE.read_text()
    html=(template.replace("__META__",json.dumps(meta,separators=(",",":")))
                  .replace("__DATA__",encoded))
    OUTPUT.write_text(html)
    print("saved",OUTPUT,"bytes",OUTPUT.stat().st_size,flush=True)


if __name__=="__main__":
    main()
