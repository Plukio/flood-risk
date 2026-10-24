"""Check whether fitted report scores actually depend on nearby physics cells."""
from __future__ import annotations

import json

import numpy as np
import torch

from flood_citywide_history_cnn_transfer import RADIUS, patches
from flood_citywide_history_ranker import SHAPE, TEST_START, load_grid
from flood_citywide_route_pretrain import lidar_static, terrain
from flood_sfincs_report_features import OUT_NPZ
from flood_sfincs_report_mask_transfer import (
    OUT_JSON, OUT_MODEL, FrozenPhysicsReportCNN, make_context,
    physics_for_fold,
)

OUT = OUT_JSON.with_name("q4_2025_sfincs_frozen_spatial_neighbor_audit.json")


def main():
    torch.set_num_threads(4)
    report=json.loads(OUT_JSON.read_text())
    states=torch.load(OUT_MODEL,map_location="cpu",weights_only=True)["states"]
    data=load_grid()
    elevation,_,_,_=terrain(data)
    with np.load(OUT_NPZ) as saved:
        raw=saved["features"]
        coverage=saved["coverage"]
    flat=data["candidate_flat"]
    district=data["district_id"].ravel()[flat]
    covered=coverage.ravel()[flat]
    rows=[]
    for fold in report["folds"]:
        fid=fold["fold"]
        ids={d["code"] for d in data["districts"] if d["code"] in
             fold["heldout_district_codes"]}
        source=np.array([(not covered[j] or data["districts"][district[j]]["code"] not in ids)
                         and covered[j] for j in range(len(flat))])
        target=(~source)&covered
        source_grid=np.zeros(SHAPE,bool)
        source_grid.ravel()[flat]=source
        static,_=lidar_static(data,source,elevation)
        physics,_=physics_for_fold(raw,coverage,source_grid)
        context_fn=make_context(physics)
        model=FrozenPhysicsReportCNN()
        model.load_state_dict(states[fid]["depth_direction_hidden"])
        model.eval()
        chosen=np.flatnonzero(target)
        rng=np.random.default_rng(917+fid)
        selected=rng.choice(chosen,size=min(128,len(chosen)),replace=False)
        image,_,_=context_fn(data,TEST_START,static)
        batch=patches(image,flat,selected)
        with torch.no_grad():
            original=model.patch_scores(batch)
            without_neighbors=batch.clone()
            own=without_neighbors[:,20:33,RADIUS,RADIUS].clone()
            without_neighbors[:,20:33]=0
            without_neighbors[:,20:33,RADIUS,RADIUS]=own
            no_neighbor=model.patch_scores(without_neighbors)
            no_vector=batch.clone()
            no_vector[:,22:25]=0
            vector_off=model.patch_scores(no_vector)
        differentiable=batch.detach().requires_grad_(True)
        model.patch_scores(differentiable).sum().backward()
        gradient=differentiable.grad
        # Row increases south. Positive u runs east, positive v runs north.
        # An incoming-versus-outgoing directional response would put greater
        # weight on positive u at the west than east neighbor, and positive v
        # at the south than north neighbor. This is a diagnostic, not a rule.
        eastward_contrast=(gradient[:,22,RADIUS,RADIUS-1]-
                           gradient[:,22,RADIUS,RADIUS+1])
        northward_contrast=(gradient[:,23,RADIUS+1,RADIUS]-
                            gradient[:,23,RADIUS-1,RADIUS])
        rows.append({"fold":fid,"sampled_heldout_cells":len(selected),
                     "mean_abs_score_change_remove_neighbor_physics":float(
                         (original-no_neighbor).abs().mean()),
                     "fraction_score_changed_remove_neighbor_physics":float(
                         ((original-no_neighbor).abs()>1e-6).float().mean()),
                     "mean_abs_score_change_remove_vector_layers":float(
                         (original-vector_off).abs().mean()),
                     "fraction_score_changed_remove_vector_layers":float(
                         ((original-vector_off).abs()>1e-6).float().mean()),
                     "mean_eastward_incoming_minus_outgoing_gradient":float(
                         eastward_contrast.mean()),
                     "mean_northward_incoming_minus_outgoing_gradient":float(
                         northward_contrast.mean()),
                     "fraction_eastward_contrast_positive":float(
                         (eastward_contrast>0).float().mean()),
                     "fraction_northward_contrast_positive":float(
                         (northward_contrast>0).float().mean())})
    result={"meaning":"Model sensitivity on 128 sampled held-out 250 m cells per fold on first Q4 day. It shows whether neighboring physics/vector input changes scores and whether gradient signs distinguish incoming from outgoing flow; it does not measure outcome capture or prove correct hydrology.",
            "rows":rows,
            "pooled_mean_neighbor_change":float(np.mean([
                r["mean_abs_score_change_remove_neighbor_physics"] for r in rows])),
            "pooled_mean_vector_change":float(np.mean([
                r["mean_abs_score_change_remove_vector_layers"] for r in rows]))}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
