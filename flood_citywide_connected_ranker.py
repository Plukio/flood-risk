"""Test canal-connected complaint history against equal-reach distance smoothing.

Weights are selected on Jan–Sep 2025. Q4 is scored after selection. OSM canal
connectivity is a proxy, not a drainage or hydraulic flow network.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import gaussian_filter

from flood_citywide_canal_graph import (
    AUDIT_FILE as RASTER_AUDIT, load as load_raster_graph, propagate,
)
from flood_citywide_true_canal_graph import (
    AUDIT_FILE as OSM_AUDIT, load as load_osm_graph,
)
from flood_citywide_history_ranker import (
    OUT_JSON as HISTORY_JSON, TEST_START, VALID_START, N_DAYS,
    capture, day_features, load_grid,
)
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval
from flood_open_weak_bangkok_holdout_eval import selected


OUT_JSON=OUT/"q4_2025_canal_connected_ranker.json"
WEIGHTS=(.2,.4,.8,1.2)
GRAPH_STEPS=(8,16,64,128)
GAUSS_SIGMAS=(4,8,16)


def names():
    return (["history_365d","recent_nearby"]+
            [f"canal_{steps}_{weight:g}" for steps in GRAPH_STEPS for weight in WEIGHTS]+
            [f"distance_{sigma}_{weight:g}" for sigma in GAUSS_SIGMAS for weight in WEIGHTS])


def scores_for_day(data:dict,graph:dict,day:int):
    _,h365,h90,near365=day_features(data,day)
    counts=(data["prefix"][day]-data["prefix"][day-365]).astype(np.float32)
    base=np.log1p(h365)+.4*np.log1p(h90)+.4*np.log1p(near365)
    connected=propagate(graph,h365,GRAPH_STEPS)
    wide={sigma:np.log1p(gaussian_filter(counts,sigma,mode="constant")
                             [data["candidate"]]) for sigma in GAUSS_SIGMAS}
    result={"history_365d":h365,"recent_nearby":base}
    for steps in GRAPH_STEPS:
        for weight in WEIGHTS:
            result[f"canal_{steps}_{weight:g}"]=base+weight*connected[steps]
    for sigma in GAUSS_SIGMAS:
        for weight in WEIGHTS:
            result[f"distance_{sigma}_{weight:g}"]=base+weight*wide[sigma]
    return result


def evaluate(data:dict,graph:dict,days:range,keep_scores:bool=False):
    candidate_flat=data["candidate_flat"]
    new=data["new"].ravel()[candidate_flat]
    names_all=names()
    positives=[]
    hits={scope:{name:[] for name in names_all} for scope in ("all","new")}
    saved={name:[] for name in names_all} if keep_scores else None
    truth=[]
    for day in days:
        scores=scores_for_day(data,graph,day)
        label=data["reports"][day].ravel()[candidate_flat]
        positives.append((int(label.sum()),int(label[new].sum())))
        if keep_scores:
            truth.append(label)
        for name,score in scores.items():
            if not np.isfinite(score).all():
                raise ValueError(f"Nonfinite {name} score on day {day}")
            hits["all"][name].append(float(np.dot(selected(score,.1),label)))
            hits["new"][name].append(float(np.dot(selected(score[new],.1),label[new])))
            if keep_scores:
                saved[name].append(score.astype(np.float32))
        if day%90==0:
            print("evaluated day index",day,flush=True)
    positives=np.asarray(positives,np.int64)
    result={}
    for i,scope in enumerate(("all","new")):
        total=int(positives[:,i].sum())
        result[scope]={}
        for name in names_all:
            captured=np.asarray(hits[scope][name],np.float64)
            result[scope][name]={"positive_cell_days":total,
                                 "candidate_cells":int(len(candidate_flat) if scope=="all" else new.sum()),
                                 "top_10pct_capture":round(float(captured.sum()/total),5) if total else None,
                                 "daily_positive":positives[:,i].tolist(),
                                 "daily_captured":np.round(captured,5).tolist()}
    return result,({k:np.stack(v) for k,v in saved.items()} if saved else None),(
        np.stack(truth) if truth else None)


def main(graph_type:str="osm_nodes"):
    OUT.mkdir(parents=True,exist_ok=True)
    data=load_grid()
    graph_loader,graph_audit_path=(
        (load_osm_graph,OSM_AUDIT) if graph_type=="osm_nodes" else
        (load_raster_graph,RASTER_AUDIT))
    graph=graph_loader(data["candidate"])
    graph_audit=json.loads(graph_audit_path.read_text())
    print("graph",graph_audit,flush=True)
    validation,_,_=evaluate(data,graph,range(VALID_START,TEST_START))
    best_canal=max((n for n in names() if n.startswith("canal_")),
                   key=lambda n:validation["all"][n]["top_10pct_capture"])
    best_distance=max((n for n in names() if n.startswith("distance_")),
                      key=lambda n:validation["all"][n]["top_10pct_capture"])
    print("validation best canal",best_canal,
          validation["all"][best_canal]["top_10pct_capture"],
          "best distance",best_distance,
          validation["all"][best_distance]["top_10pct_capture"],flush=True)
    q4,scores,truth=evaluate(data,graph,range(TEST_START,N_DAYS),keep_scores=True)
    history=json.loads(HISTORY_JSON.read_text())
    if (q4["all"]["history_365d"]["top_10pct_capture"] !=
        history["q4_scopes"]["citywide"]["history_365d"]["top_10pct_capture"] or
        q4["all"]["recent_nearby"]["top_10pct_capture"] !=
        history["q4_scopes"]["citywide"]["history_recent_nearby"]["top_10pct_capture"] or
        q4["new"]["recent_nearby"]["top_10pct_capture"] !=
        history["q4_scopes"]["outside_750m_training_buffer"]["history_recent_nearby"]["top_10pct_capture"]):
        raise ValueError("Connected experiment comparison baseline changed")
    summary={}
    for scope in ("all","new"):
        rows=q4[scope]
        summary[scope]={name:rows[name] for name in
                        ("history_365d","recent_nearby",best_canal,best_distance)}
        summary[scope]["canal_minus_recent_nearby_95pct_weekly_block_interval"]=(
            paired_interval(rows[best_canal],rows["recent_nearby"]))
        summary[scope]["canal_minus_distance_95pct_weekly_block_interval"]=(
            paired_interval(rows[best_canal],rows[best_distance]))
    candidate_flat=data["candidate_flat"]
    new=data["new"].ravel()[candidate_flat]
    district_id=data["district_id"].ravel()[candidate_flat]
    districts={}
    for i,district in enumerate(data["districts"]):
        mask=district_id==i
        districts[district["code"]]={"name":district["name_en"],
            "q4":{name:capture(scores[name],truth,mask) for name in
                  ("recent_nearby",best_canal,best_distance)},
            "q4_new":{name:capture(scores[name],truth,mask & new) for name in
                  ("recent_nearby",best_canal,best_distance)}}
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "graph_type":graph_type,
            "graph_audit":graph_audit,"validation_period":["2025-01-01","2025-09-30"],
            "q4_period":["2025-10-01","2025-12-31"],
            "base":"log1p(prior365) + 0.4 log1p(prior90) + 0.4 log1p(250m Gaussian prior365); no rain",
            "canal_feature":("OSM shared-node canal graph with intermediate nodes on segments over 250 m" if graph_type=="osm_nodes" else "OSM canal raster at 250 m; crossings in the same cell may merge")+"; report histories projected to nearest canal node within 1250m, row-normalized graph walk 8, 16, 64, or 128 steps, area-normalized mean projected back; no direction or gate/pump state",
            "distance_control":"Gaussian smoothing of prior365 count with sigma 4, 8, or 16 cells (1, 2, or 4 km), no topology",
            "selected_canal_recipe":best_canal,"selected_distance_recipe":best_distance,
            "selection_rule":"Best capture within each feature family on Jan-Sep 2025 citywide; the graph feature is not promoted if it trails the base",
            "validation_citywide_capture":{n:r["top_10pct_capture"] for n,r in validation["all"].items()},
            "validation_new_capture":{n:r["top_10pct_capture"] for n,r in validation["new"].items()},
            "q4_summary":summary,"q4_districts":districts,
            "limitations":"Canal topology is not verified drainage. Q4 was inspected previously. Reports are positive observations; unreported is unknown. These results are not hydraulic simulations."}
    output=OUT/("q4_2025_canal_connected_ranker_osm_nodes.json" if graph_type=="osm_nodes"
                else "q4_2025_canal_connected_ranker.json")
    output.write_text(json.dumps(result,indent=2)+"\n")
    print("Q4",{scope:{n:r["top_10pct_capture"] for n,r in rows.items() if isinstance(r,dict)}
                 for scope,rows in summary.items()},flush=True)
    print("saved",output,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--graph",choices=("osm_nodes","raster"),default="osm_nodes")
    main(parser.parse_args().graph)
