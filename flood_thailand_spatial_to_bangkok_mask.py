"""Transfer geographically tested synthetic-flow filters to Bangkok reports.

The common SRTM/OSM/WorldCover input channels and spatial steps come from the
Thai multi-place pretrain. This fits the usual strict history-masked report
correction afterward. The pretraining target is synthetic, never flood truth.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_citywide_drainage_adapter import BMA_JSON, BMA_MODEL
from flood_citywide_drainage_network import FEATURES
from flood_citywide_drainage_spatial_pretrain import fit_reports, score_days
from flood_citywide_history_cnn_transfer import SEED
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import ContextResidualCNN, pool
from flood_citywide_isolated_group_mask import mask_self_check
from flood_citywide_physics_pretrain_transfer import SurfaceContextCNN20
from flood_citywide_route_pretrain import lidar_static, terrain
from flood_citywide_surface_osm import build as buildings
from flood_multi_target_citywide_data import OUT, ROWS, COLS
from flood_thailand_spatial_physics_pretrain import COMMON, OUT_MODEL as PRETRAIN_MODEL


OUT_JSON = OUT / "q4_2025_thai_spatial_to_bangkok_mask_fivefold.json"
OUT_MODEL = OUT / "q4_2025_thai_spatial_to_bangkok_mask_fivefold.pt"
ALPHAS = (0., .1, .25, .5, 1.)
ARMS = ("scratch", "thai_spatial_pretrained")


def report_template(data, drainage, building, normalization):
    """Use the pretrain's common channels with the same 20 km altitude center."""
    raw = np.zeros((10, *SHAPE), np.float32)
    raw[:, data["candidate"]] = data["static"][:, list(COMMON)].T
    city_median = float(np.median(raw[0, data["candidate"]]))
    for r in range(ROWS):
        for c in range(COLS):
            ys = slice(r*80, (r+1)*80)
            xs = slice(c*80, (c+1)*80)
            candidate = data["candidate"][ys, xs]
            median = float(np.median(raw[0, ys, xs][candidate])) if candidate.any() else city_median
            raw[0, ys, xs] -= median
    mean = np.asarray(normalization["mean"], np.float32)[:, None, None]
    std = np.asarray(normalization["std"], np.float32)[:, None, None]
    image = np.zeros((20, *SHAPE), np.float32)
    image[2] = .5*data["candidate"]
    image[3:13] = np.clip((raw-mean)/std, -5, 5)*data["candidate"]
    image[13] = data["candidate"]
    image[14:18] = drainage[[0, 1, 3, 4]]
    image[18] = building
    image[19] = 1
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite report image")
    return image


def transfer_initial(seed, pretrain_state):
    # Keep report-history and Bangkok-only channel initialization matched to
    # scratch; copy only shared static-channel filters and spatial steps.
    torch.manual_seed(seed)
    model = SurfaceContextCNN20()
    with torch.no_grad():
        model.entry.weight[:, 3:13].copy_(pretrain_state["entry.weight"][:, 3:13])
    keys = model.steps.state_dict().keys()
    model.steps.load_state_dict({k: pretrain_state["steps."+k] for k in keys})
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    suffix = f"_seed_{args.seed_offset}" if args.seed_offset else ""
    pretrain_path = PRETRAIN_MODEL.with_name(PRETRAIN_MODEL.stem+suffix+".pt")
    prior = torch.load(pretrain_path, map_location="cpu", weights_only=True)
    pretrain_state = prior["spatial"]
    normalization = prior["normalization"]
    data = load_grid()
    drainage = np.load(FEATURES)["features"]
    building = buildings()
    raw_elevation, _, _, _ = terrain(data)
    template = report_template(data, drainage, building, normalization)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(50), 5)
    previous = json.loads(BMA_JSON.with_name(BMA_JSON.stem+suffix+".json").read_text())
    previous_states = torch.load(BMA_MODEL.with_name(BMA_MODEL.stem+suffix+".pt"),
                                 map_location="cpu", weights_only=True)["states"]
    records, states = [], {}
    for fold_id, heldout in enumerate(folds, 1):
        source = ~np.isin(district, heldout)
        static, _ = lidar_static(data, source, raw_elevation)
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
        record = {"fold": fold_id,
                  "heldout_district_codes": old["heldout_district_codes"],
                  "frozen_bma_alpha": base_alpha, "arms": {}}
        states[fold_id] = {}
        for arm, initial in (("scratch", None),
                             ("thai_spatial_pretrained", transfer_initial(seed, pretrain_state))):
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
            states[fold_id][arm] = {k: v.cpu() for k, v in model.state_dict().items()}
            print("seed", args.seed_offset, "fold", fold_id, arm, "alpha", alpha,
                  "cells/groups", test[str(alpha)]["isolated_hits"],
                  test[str(alpha)]["group_hits"], flush=True)
        records.append(record)
    pooled = {}
    for arm in ARMS:
        pooled[arm] = {str(a): pool([f["arms"][arm]["heldout_q4"][str(a)]
                                     for f in records]) for a in ALPHAS}
        pooled[arm]["source_selected"] = pool([
            f["arms"][arm]["heldout_q4"][str(f["arms"][arm]["selected_alpha"])]
            for f in records])
        expected = previous["pooled_heldout_q4"]["bma_real"]["source_selected"]
        base = pooled[arm]["0.0"]
        if (abs(base["isolated_hits"]-expected["isolated_hits"])>.01 or
            abs(base["group_hits"]-expected["group_hits"])>.01 or
            base["isolated_positive"] != 332 or base["isolated_groups"] != 158):
            raise ValueError("Frozen BMA comparator or denominators changed")
    result = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Do spatial filters learned from synthetic flow across Thai places help the masked Bangkok report queue?",
        "target": "Q4 2025 Traffy report-positive 250m cells, not verified flooding",
        "pretraining_model": str(pretrain_path),
        "transfer": "Shared SRTM/OSM/WorldCover entry weights and five spatial steps; report, Bangkok-only and bias weights initialized as scratch; all fine-tuned",
        "report_training": "2023-24 source districts; exact target-neighborhood history mask; group-level loss",
        "selection": "Jan-Sep 2025 source districts only; group then cell capture",
        "queue": "Daily top 10% candidate-cell budget with 20% isolated reserve",
        "seed_offset": args.seed_offset, "folds": records,
        "pooled_heldout_q4": pooled,
        "limitations": ["The pretraining teacher is synthetic and uncalibrated",
                        "SRTM pretraining and Bangkok LiDAR main model use different elevation products",
                        "Unreported cells are unknown, not verified dry",
                        "Q4 2025 has been repeatedly inspected; exploratory"]}
    OUT_JSON.with_name(OUT_JSON.stem+suffix+".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    torch.save({"states": states}, OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt"))
    print("pooled", {arm:(r["source_selected"]["isolated_hits"],
                         r["source_selected"]["group_hits"])
                     for arm,r in pooled.items()}, flush=True)


if __name__ == "__main__":
    main()
