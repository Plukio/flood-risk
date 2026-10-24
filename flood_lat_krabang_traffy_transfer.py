"""Transfer the Lat Krabang physical scenario score to 2025 Traffy queues.

The saved five-fold masked CNN remains frozen. All three controlled rain
scenarios and ordinary/reversed controls are evaluated, without importing the
2026 GISTDA-selected scenario into the earlier 2025 report backtest.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy.stats import rankdata

from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, VALID_START, load_grid
from flood_citywide_spatial_mask_augmentation import OUT_MODEL
from flood_citywide_spatial_novel_queue import (SpatialNovelCNN, day_maps,
                                                 queue_weights, static_image,
                                                 top_weights)
from flood_lat_krabang_edge_flow_pilot import (CELL_M, OUT as FLOW_OUT,
                                                SCENARIOS, grid_data, simulate)
from flood_multi_target_citywide_data import X0, Y0, ROWS, TILE_M
from flood_open_weak_data import FEATURE_NAMES


RESULT = FLOW_OUT / "traffy_2025_transfer.json"
ALPHAS = (0., .25, .5, 1.)
QUOTA = .20
CONTEXT = 8  # CNN radius is 7 cells


def scope_and_scores(data: dict, terrain: dict):
    district_ix = next(i for i, d in enumerate(data["districts"])
                       if d["code"] == "1011")
    flat = data["candidate_flat"]
    rr, cc = np.unravel_index(flat, SHAPE)
    west = terrain["transform"].c
    north = terrain["transform"].f
    east = west + terrain["n"] * CELL_M
    south = north - terrain["n"] * CELL_M
    cell_west = X0 + cc * CELL_M
    cell_north = Y0 + ROWS * TILE_M - rr * CELL_M
    district = data["district_id"].ravel()[flat]
    scope = ((district == district_ix) & (cell_west >= west) &
             (cell_west + CELL_M <= east) & (cell_north <= north) &
             (cell_north - CELL_M >= south))
    ids = np.flatnonzero(scope)
    if len(ids) < 100:
        raise ValueError("Too few fully covered Lat Krabang report cells")
    source_transform = Affine(CELL_M, 0, west, 0, -CELL_M, north)
    destination_transform = Affine(CELL_M, 0, X0, 0, -CELL_M, Y0 + ROWS * TILE_M)
    scenario_scores = {}
    for mm, hours in SCENARIOS:
        for arm in ("ordinary", "saddle", "reversed"):
            output = simulate(terrain, mm, hours, arm)
            source = output["wet_fraction_gt_10cm"].reshape(terrain["n"], terrain["n"])
            destination = np.full(SHAPE, -999., np.float32)
            reproject(source.astype(np.float32), destination,
                      src_transform=source_transform, src_crs="EPSG:32647",
                      src_nodata=-999, dst_transform=destination_transform,
                      dst_crs="EPSG:32647", dst_nodata=-999,
                      resampling=Resampling.average)
            values = destination.ravel()[flat][ids]
            if np.any(values < 0) or np.any(values > 1):
                raise ValueError(f"Physical score/grid mismatch: {mm}, {hours}, {arm}")
            scenario_scores[f"{mm}mm_{hours}h_{arm}"] = values
    return ids, rr[ids], cc[ids], district_ix, scenario_scores


def percentile(value):
    return rankdata(value, method="average").astype(np.float32) / len(value)


def evaluate_day(base, simple, cnn, novel, truth, physical):
    budget = .1 * len(base)
    weights = {
        "fixed_history": top_weights(base, budget),
        "nearby_history_novel": queue_weights(base, simple, novel, QUOTA),
        "masked_cnn": queue_weights(base, cnn, novel, QUOTA)}
    for name, value in physical.items():
        weights[name] = queue_weights(base, value, novel, QUOTA)
        for alpha in ALPHAS[1:-1]:
            # Rank scales avoid mistaking metres or pixel fractions for logits.
            blended = ((1-alpha)*percentile(cnn) +
                       alpha*percentile(value))
            weights[f"{name}_blend_{alpha:g}"] = queue_weights(
                base, blended, novel, QUOTA)
    result = {"positive": int(truth.sum()), "novel_positive": int(truth[novel].sum())}
    for name, weight in weights.items():
        result[name] = (float(np.dot(weight, truth)),
                        float(np.dot(weight[novel], truth[novel])))
    return result


def aggregate(daily):
    positives = sum(row["positive"] for row in daily)
    novel = sum(row["novel_positive"] for row in daily)
    names = [key for key in daily[0] if key not in ("positive", "novel_positive")]
    return {"candidate_cell_days": len(daily),
            "report_positive_cell_days": positives,
            "first_in_prior_year_positive_cell_days": novel,
            "arms": {name: {
                "top_10pct_report_capture": round(sum(row[name][0] for row in daily)/positives, 5)
                if positives else None,
                "novel_capture": round(
                    sum(row[name][1] for row in daily)/novel, 5) if novel else None,
                "captured_positive_cell_days": round(sum(row[name][0] for row in daily), 3),
                "captured_novel_cell_days": round(sum(row[name][1] for row in daily), 3)}
                for name in names}}


def main():
    torch.set_num_threads(4)
    terrain = grid_data()
    data = load_grid()
    ids, rr, cc, district_ix, physical = scope_and_scores(data, terrain)
    districts = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    fold = next(i for i, part in enumerate(districts, 1) if district_ix in part)
    source = ~np.isin(data["district_id"].ravel()[data["candidate_flat"]],
                     districts[fold - 1])
    static, _ = static_image(data, source)
    states = torch.load(OUT_MODEL, map_location="cpu", weights_only=True)["states"]
    model = SpatialNovelCNN().eval()
    model.load_state_dict(states[fold]["synthetic_center_block"])
    r0, r1 = max(0, rr.min()-CONTEXT), min(SHAPE[0], rr.max()+CONTEXT+1)
    c0, c1 = max(0, cc.min()-CONTEXT), min(SHAPE[1], cc.max()+CONTEXT+1)
    if min(rr.min()-r0, r1-1-rr.max(), cc.min()-c0, c1-1-cc.max()) < 7:
        raise ValueError("Insufficient CNN context")
    daily_validation, daily_test = [], []
    flat = data["candidate_flat"]
    crop_fraction = data["static"][ids, FEATURE_NAMES.index(
        "worldcover_crop_fraction_2021")]
    built_fraction = data["static"][ids, FEATURE_NAMES.index(
        "worldcover_built_fraction_2021")]
    cover_audit = {"area_crop_fraction_mean": round(float(crop_fraction.mean()), 5),
                   "area_built_fraction_mean": round(float(built_fraction.mean()), 5),
                   "q4_positive_crop_fraction_sum": 0.,
                   "q4_positive_built_fraction_sum": 0.,
                   "q4_positive_cells_majority_crop": 0,
                   "q4_positive_cells_majority_built": 0}
    for day in range(VALID_START, N_DAYS):
        image, base_all, simple_all, novel_all = day_maps(data, day, static)
        with torch.no_grad():
            local = model(torch.from_numpy(image[None, :, r0:r1, c0:c1]))[0]
            cnn = local[rr-r0, cc-c0].numpy()
            if day == VALID_START:
                full = model(torch.from_numpy(image[None]))[0, rr, cc].numpy()
                if not np.allclose(cnn, full, atol=1e-5):
                    raise ValueError("Cropped CNN differs from frozen full-grid model")
        truth = data["reports"][day].ravel()[flat][ids]
        if day >= TEST_START:
            cover_audit["q4_positive_crop_fraction_sum"] += float(np.dot(
                crop_fraction, truth))
            cover_audit["q4_positive_built_fraction_sum"] += float(np.dot(
                built_fraction, truth))
            cover_audit["q4_positive_cells_majority_crop"] += int((truth[
                crop_fraction > .5]).sum())
            cover_audit["q4_positive_cells_majority_built"] += int((truth[
                built_fraction > .5]).sum())
        row = evaluate_day(base_all[ids], simple_all[ids], cnn,
                           novel_all[ids], truth, physical)
        (daily_validation if day < TEST_START else daily_test).append(row)
        if day in (VALID_START, TEST_START, N_DAYS-1):
            print("scored day", day, "positive", row["positive"], flush=True)
    validation = aggregate(daily_validation)
    q4 = aggregate(daily_test)
    for key in ("crop", "built"):
        cover_audit[f"q4_positive_{key}_fraction_mean"] = round(
            cover_audit.pop(f"q4_positive_{key}_fraction_sum") /
            q4["report_positive_cell_days"], 5)
    # Choose blend only with January-September Traffy labels. Scenario arms are
    # reported individually; no 2026 GISTDA selection is carried into 2025.
    base_val = validation["arms"]["masked_cnn"]["novel_capture"]
    candidate_names = ["masked_cnn"] + [key for key in validation["arms"]
                                         if key.endswith(("blend_0.25", "blend_0.5"))]
    selected = max(candidate_names, key=lambda name: (
        validation["arms"][name]["novel_capture"] or -1,
        -candidate_names.index(name)))
    summary = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": "Does a 10 m terrain surface-flow scenario score improve a frozen masked-CNN Traffy report queue in its Lat Krabang DTM window?",
        "area": "Fully covered 250 m cells assigned to Lat Krabang inside the 8.5 km DTM crop",
        "candidate_cells": int(len(ids)), "district_fold": fold,
        "model": "Saved five-fold synthetic-center-block masked CNN; Lat Krabang excluded from weight fitting",
        "validation_dates": "2025-01-01 through 2025-09-30",
        "test_dates": "2025-10-01 through 2025-12-31",
        "queue": "Daily top 10% of local cells; CNN/flow/nearby arms reserve 20% of slots for no-own-report-in-prior-365-days cells; unused slots spill over. Fixed-history control ranks all cells without a reserve.",
        "physical_scores": "Fixed controlled rain scenarios; fraction of 10 m subpixels over 10 cm simulated depth, areal resampled to report cells",
        "2026_gistda_scenario_selection_used": False,
        "source_selected_blend": selected,
        "source_masked_cnn_novel_capture": base_val,
        "worldcover_target_mix": cover_audit,
        "validation": validation, "q4": q4,
        "notes": ["Only the DTM-covered part of one district is tested, so values are not comparable with the five-fold citywide 14% capture",
                  "Traffy absence is unreported, not verified dry",
                  "Terrain acquisition date is not confirmed for a strict 2025 forecast backtest",
                  "Static controlled rainfall is not matched to report-day rain or drainage operations",
                  "Q4 2025 has been inspected repeatedly in related project experiments"]}
    RESULT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"cells": len(ids), "validation_novel": validation[
        "first_in_prior_year_positive_cell_days"], "q4_novel": q4[
        "first_in_prior_year_positive_cell_days"], "selected": selected,
        "q4_capture": {key: value["novel_capture"]
                       for key, value in q4["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
