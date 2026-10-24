"""Conservative synthetic-flow pretraining and held-out report transfer.

This is a 250 m prototype. Continuity and passive-head direction are exact
model constraints; teacher rates are heuristic and not observed inundation.
The report comparison keeps the frozen BMA-road CNN and its strict queue.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn

from flood_citywide_conservative_flow import (
    ConservativeFlowCNN, physics_self_check, scenario, teacher_step,
)
from flood_citywide_drainage_adapter import BMA_JSON, BMA_MODEL
from flood_citywide_drainage_network import AUDIT as DRAINAGE_AUDIT, FEATURES
from flood_citywide_drainage_spatial_pretrain import fit_reports, score_days
from flood_citywide_history_cnn_transfer import SEED
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import ContextResidualCNN, pool
from flood_citywide_isolated_group_mask import mask_self_check
from flood_citywide_route_pretrain import lidar_static, terrain
from flood_citywide_spatial_novel_queue import SpatialNovelCNN
from flood_citywide_surface_osm import AUDIT as BUILDING_AUDIT, build as buildings
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_conservative_physics_pretrain_fivefold.json"
OUT_MODEL = OUT / "q4_2025_conservative_physics_pretrain_fivefold.pt"
PRETRAIN_STEPS = 120
ALPHAS = (0., .1, .25, .5, 1.)
ARMS = ("scratch", "physics_pretrained")
LOSS_WEIGHTS = {"surface": 1., "pipe": .4, "infiltration": .15,
                "outlet": .15, "flow": .25}


class SurfaceContextCNN20(SpatialNovelCNN):
    """Report correction with weights transferable from the flow trunk."""

    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(20, 8, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


def template_image(data, static, drainage, building):
    image = np.zeros((20, *SHAPE), np.float32)
    image[2] = .5*data["candidate"]  # unknown historical gate state
    image[3:13] = static
    image[13] = data["candidate"]
    image[14:18] = drainage[[0, 1, 3, 4]]
    image[18] = building
    image[19] = 1
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite physics/report image")
    return image


def teacher_parameters(data, building, drainage):
    candidate = data["candidate"]
    raw = np.zeros((13, *SHAPE), np.float32)
    raw[:, candidate] = data["static"].T
    cover = np.stack((np.clip(raw[7], 0, 1),
                      np.clip(raw[9], 0, 1))).astype(np.float32)
    roads = np.clip(raw[5]/7., 0, 1)
    return cover, roads, drainage[[0, 1, 3]]


def prepare_scenario(rng, template, candidate, pipe_prox, elevation):
    surface, rain, pipe, gate = scenario(rng, candidate, pipe_prox)
    available = surface+rain
    image = template.copy()
    image[0] = np.log1p(1000*available)
    image[1] = np.log1p(1000*pipe)
    image[2] = gate
    return image, available, pipe, gate


def target_tensors(teacher):
    return {key: torch.from_numpy(np.ascontiguousarray(value[None, None]
                      if key != "flow" else value[None]))
            for key, value in teacher.items()
            if key in LOSS_WEIGHTS}


def loss_terms(pred, truth, mask):
    wet = mask & (truth["surface"] > .001)
    if not torch.any(wet):
        raise ValueError("No wet cells in synthetic source region")
    terms = {}
    for key, weight in LOSS_WEIGHTS.items():
        delta = (pred[key]-truth[key])*1000
        if key == "flow":
            selected = wet.expand_as(delta)
        else:
            selected = wet
        terms[key] = weight*torch.nn.functional.smooth_l1_loss(
            delta[selected], torch.zeros_like(delta[selected]))
    return sum(terms.values()), {key: float(value.detach())
                                  for key, value in terms.items()}


def synthetic_eval(model, data, template, source, elevation, cover, roads,
                   drainage, seed):
    rng = np.random.default_rng(seed+92057)
    scope = np.zeros(SHAPE, bool)
    scope.ravel()[data["candidate_flat"]] = ~source
    model_mae, persistence_mae, continuity = [], [], []
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            image, available, pipe, gate = prepare_scenario(
                rng, template, data["candidate"], drainage[0], elevation)
            teacher = teacher_step(available*0, available, pipe, gate,
                                   elevation, data["candidate"], cover,
                                   template[18], roads, drainage)
            prediction = model(torch.from_numpy(image[None]),
                               torch.from_numpy(available[None,None]),
                               torch.from_numpy(pipe[None,None]),
                               torch.from_numpy(elevation[None,None]))
            predicted = prediction["surface"][0,0].numpy()
            wet = scope & (teacher["surface"]>.001)
            model_mae.append(float(np.mean(np.abs(predicted[wet]-
                                                 teacher["surface"][wet]))*1000))
            persistence_mae.append(float(np.mean(np.abs(available[wet]-
                                                       teacher["surface"][wet]))*1000))
            got = sum(float(prediction[key].sum()) for key in
                      ("surface", "pipe", "infiltration", "outlet"))
            expected = float((available+pipe).sum())
            continuity.append(abs(got-expected)*62500.)
    return {"heldout_surface_mae_mm": round(float(np.mean(model_mae)), 4),
            "persistence_mae_mm": round(float(np.mean(persistence_mae)), 4),
            "whole_grid_balance_error_m3_max": round(max(continuity), 4)}


def pretrain(data, source, template, elevation, cover, roads, drainage, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+491)
    model = ConservativeFlowCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    scope = np.zeros(SHAPE, bool)
    scope.ravel()[data["candidate_flat"]] = source
    scope = torch.from_numpy(scope[None,None])
    z = torch.from_numpy(elevation[None,None])
    losses = []
    for step in range(PRETRAIN_STEPS):
        image, available, pipe, gate = prepare_scenario(
            rng, template, data["candidate"], drainage[0], elevation)
        # The teacher sees the pre-rain surface as zero here; each sample is
        # a one-step rainfall pulse plus randomized pipe storage.
        teacher = teacher_step(np.zeros(SHAPE, np.float32), available,
                               pipe, gate, elevation, data["candidate"],
                               cover, template[18], roads, drainage)
        truth = target_tensors(teacher)
        prediction = model(torch.from_numpy(image[None]),
                           torch.from_numpy(available[None,None]),
                           torch.from_numpy(pipe[None,None]), z)
        loss, pieces = loss_terms(prediction, truth, scope)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 30 == 0:
            print("physics step", step+1, "loss", round(float(np.mean(losses[-30:])),5),
                  "terms", {k:round(v,4) for k,v in pieces.items()}, flush=True)
    model.eval()
    quality = synthetic_eval(model, data, template, source, elevation,
                             cover, roads, drainage, seed)
    return model, {"steps": PRETRAIN_STEPS,
                   "loss_first_20": round(float(np.mean(losses[:20])),5),
                   "loss_last_20": round(float(np.mean(losses[-20:])),5),
                   **quality}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    physics_self_check()
    data = load_grid()
    drainage = np.load(FEATURES)["features"]
    building = buildings()
    raw_elevation, _, _, _ = terrain(data)
    good = data["candidate"] & np.isfinite(raw_elevation)
    nearest = distance_transform_edt(~good, return_distances=False,
                                     return_indices=True)
    elevation = raw_elevation[tuple(nearest)].astype(np.float32)
    cover, roads, teacher_drainage = teacher_parameters(data, building, drainage)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(50), 5)
    suffix = f"_seed_{args.seed_offset}" if args.seed_offset else ""
    previous = json.loads(BMA_JSON.with_name(BMA_JSON.stem+suffix+".json").read_text())
    previous_states = torch.load(BMA_MODEL.with_name(BMA_MODEL.stem+suffix+".pt"),
                                 map_location="cpu", weights_only=True)["states"]
    records, states = [], {}
    for fold_id, heldout in enumerate(folds, 1):
        if args.fold is not None and fold_id != args.fold:
            continue
        source = ~np.isin(district, heldout)
        static, normalization = lidar_static(data, source, raw_elevation)
        template = template_image(data, static, drainage, building)
        mask_self_check(data, static)
        old = next(f for f in previous["folds"] if f["fold"] == fold_id)
        if old["heldout_district_codes"] != [data["districts"][j]["code"]
                                              for j in heldout]:
            raise ValueError("Frozen BMA fold differs")
        frozen = ContextResidualCNN()
        frozen.load_state_dict(previous_states[fold_id]["bma_real"])
        frozen.eval()
        base_alpha = float(old["arms"]["bma_real"]["selected_alpha"])
        seed = SEED+fold_id*1000+args.seed_offset
        print("fold", fold_id, "source", int(source.sum()), flush=True)
        physical, pretrain_audit = pretrain(data, source, template, elevation,
                                            cover, roads, teacher_drainage, seed)
        record = {"fold": fold_id,
                  "heldout_district_codes": old["heldout_district_codes"],
                  "normalization": normalization,
                  "pretraining": pretrain_audit,
                  "frozen_bma_alpha": base_alpha, "arms": {}}
        states[fold_id] = {}
        for arm, initial in (("scratch", None),
                             ("physics_pretrained", physical)):
            model, fit_audit = fit_reports(data, source, static, template,
                frozen, base_alpha, seed, initial, model_factory=SurfaceContextCNN20)
            valid = score_days(data, source, static, template, frozen,
                               base_alpha, model, range(VALID_START, TEST_START))
            alpha = max(ALPHAS, key=lambda a: (
                valid[str(a)]["isolated_group_capture"],
                valid[str(a)]["isolated_cell_capture"], -a))
            test = score_days(data, ~source, static, template, frozen,
                              base_alpha, model, range(TEST_START, N_DAYS))
            record["arms"][arm] = {"fit": fit_audit,
                                   "source_validation": valid,
                                   "selected_alpha": alpha,
                                   "heldout_q4": test}
            states[fold_id][arm] = {k: v.cpu() for k,v in model.state_dict().items()}
            print("fold", fold_id, arm, "alpha", alpha, "q4 cells/groups",
                  test[str(alpha)]["isolated_hits"],
                  test[str(alpha)]["group_hits"], flush=True)
        records.append(record)
    pooled = {}
    for arm in ARMS:
        pooled[arm] = {str(a): pool([f["arms"][arm]["heldout_q4"][str(a)]
                                     for f in records]) for a in ALPHAS}
        pooled[arm]["source_selected"] = pool([
            f["arms"][arm]["heldout_q4"][str(f["arms"][arm]["selected_alpha"])]
            for f in records])
        if args.fold is None:
            expected = previous["pooled_heldout_q4"]["bma_real"]["source_selected"]
            base = pooled[arm]["0.0"]
            if (abs(base["isolated_hits"]-expected["isolated_hits"])>.01 or
                abs(base["group_hits"]-expected["group_hits"])>.01 or
                base["isolated_positive"]!=332 or base["isolated_groups"]!=158):
                raise ValueError("Frozen BMA-road comparator changed")
    result = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "question": "Does conservative synthetic-flow pretraining of all spatial context improve held-out emerging report ranking?",
              "target": "Q4 2025 Traffy report-positive 250m cells, not observed flood extent",
              "physics_hard_rules": ["Exact transfer continuity across 4-neighbor surface and pipe-reservoir ledger",
                                     "Passive surface transfer only toward lower hydraulic head",
                                     "Nonnegative surface, pipe, infiltration and outlet volumes"],
              "teacher_assumptions": "Heuristic cover-dependent infiltration, mobility and pipe/outlet rates; randomized rain, initial water and gate openness; not calibrated hydrodynamics",
              "pretraining_scale": "One illustrative 250m cell step; not a 10m engineering solver",
              "features": ["pre-step water after rain, pipe storage, gate scenario",
                           "BMA LiDAR elevation/relative elevation/slope",
                           "OSM canal and road length/distance",
                           "WorldCover built/crop/tree-grass fractions",
                           "BMA pipe/candidate outlet/pump/gate proximity",
                           "OSM building footprint fraction, candidate/valid mask"],
              "transfer": "Copy spatial entry and five residual convolution steps, reset report head, fine-tune all weights with strict target-history mask",
              "report_training": "2023-24 source districts, same isolated group loss as matched scratch arm; frozen BMA-road score",
              "selection": "Jan-Sep 2025 source districts only; group then cell capture and smaller alpha",
              "queue": "10% daily candidate-cell budget with 20% isolated reservation",
              "physics_steps": PRETRAIN_STEPS, "physics_loss_weights": LOSS_WEIGHTS,
              "drainage_audit": json.loads(DRAINAGE_AUDIT.read_text()),
              "building_audit": json.loads(BUILDING_AUDIT.read_text()),
              "seed_offset": args.seed_offset, "folds": records,
              "pooled_heldout_q4": pooled,
              "limitations": ["Teacher flow rates are invented priors, not Bangkok observations",
                              "250m cells are too coarse for curbs, inlets and pipe hydraulics",
                              "BMA pipe and outlet links remain unverified",
                              "Unknown historical gate, pump and canal states",
                              "Unreported comparisons are unknown, not verified dry",
                              "Q4 2025 has been repeatedly inspected; exploratory"]}
    suffix += f"_fold_{args.fold}" if args.fold else ""
    OUT_JSON.with_name(OUT_JSON.stem+suffix+".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    torch.save({"states": states}, OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt"))
    print("pooled", {arm:(r["source_selected"]["isolated_hits"],
                         r["source_selected"]["group_hits"])
                     for arm,r in pooled.items()}, flush=True)


if __name__ == "__main__":
    main()
