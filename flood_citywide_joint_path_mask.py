"""Joint pre-Q4 mapped-water + target-masked report training on a shared trunk.

Four matched arms cross D8 path initialization and UNOSAT auxiliary training.
The report queue, masks, draws, and source-only alpha selection stay fixed.
UNOSAT labels are dated 2021/2024; Bangkok and Pathum Thani scenes are
excluded from auxiliary fitting. No 2026 mapped-water labels enter this test.

Run: .venv/bin/python flood_citywide_joint_path_mask.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flood_citywide_history_cnn_transfer import SEED
from flood_citywide_history_ranker import N_DAYS, TEST_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import (
    ALPHAS, ContextResidualCNN, pool, score_days,
)
from flood_citywide_isolated_group_mask import mask_self_check, train
from flood_citywide_path_pretrain_mask_transfer import (
    OUT_JSON as REPORT_ONLY_JSON, PRETRAIN_MODEL,
)
from flood_citywide_route_pretrain import OUT, lidar_static, terrain
from flood_citywide_spatial_novel_queue import STATIC_NAMES, SpatialNovelCNN
from flood_label_audit import scene_path, validate_scene
from flood_open_weak_bangkok_holdout_eval import selected
from flood_open_weak_data import FEATURE_NAMES, OUTPUT
from flood_open_weak_train import weak_loss

ARM = "isolated_plus_masked_adjacent"
SAT_INTERVAL = 2
OUT_JSON = OUT / "q4_2025_joint_unosat_path_mask_fivefold.json"
OUT_MODEL = OUT / "q4_2025_joint_unosat_path_mask_fivefold.pt"
SAT_COLUMNS = [FEATURE_NAMES.index(name) for name in STATIC_NAMES]
SAT_HOLDOUT = {"pathum_thani"}
SAT_EXCLUDE = {"nong_chok", "bang_kapi_core"}


class JointMaskedCNN(ContextResidualCNN):
    """Domain-specific satellite input/head with shared spatial steps."""

    def __init__(self):
        super().__init__()
        self.sat_entry = nn.Conv2d(len(SAT_COLUMNS)+1, 8, 1)
        self.sat_head = nn.Conv2d(8, 1, 1)

    def satellite_scores(self, image: torch.Tensor) -> torch.Tensor:
        valid = image[:, -1:]
        features = F.relu(self.sat_entry(image))*valid
        for step in self.steps:
            update = step["mix"](F.relu(step["spatial"](features)))
            features = F.relu(features+.25*update)*valid
        return self.sat_head(features)[:, 0]


def load_satellite() -> tuple[list[dict], list[dict], dict]:
    manifest = json.loads((OUTPUT/"data_audit.json").read_text())
    if tuple(manifest["features"]) != FEATURE_NAMES:
        raise ValueError("Satellite static feature order changed")
    source, holdout = [], []
    for entry in manifest["scenes"]:
        if entry["source"] not in ("unosat2021", "unosat2024"):
            continue
        if entry["day"] >= "2025-01-01" or entry["region"] in SAT_EXCLUDE:
            continue
        with np.load(scene_path(entry), allow_pickle=False) as z:
            raw = {key: z[key].astype(np.float32) for key in
                   ("static", "weather", "wet", "observed")}
        validate_scene(entry, raw, manifest)
        row = {**entry, "static": raw["static"][SAT_COLUMNS],
               "wet": raw["wet"], "observed": raw["observed"]}
        (holdout if entry["region"] in SAT_HOLDOUT else source).append(row)
    unique = {s["region"]: s["static"] for s in source}
    values = np.stack(list(unique.values()))
    mean = values.mean(axis=(0, 2, 3)).astype(np.float32)
    std = np.maximum(values.std(axis=(0, 2, 3)), .05).astype(np.float32)
    for s in source+holdout:
        static = np.clip((s["static"]-mean[:, None, None])/
                         std[:, None, None], -5, 5)
        s["image"] = np.concatenate((static, np.ones((1, 80, 80), np.float32)))
    if len(source) < 30 or len(holdout) != 4 or any(
            s["day"] >= "2025-01-01" for s in source):
        raise ValueError("Pre-Q4 UNOSAT training/holdout changed")
    audit = {"training_scenes": len(source), "training_regions": len(unique),
             "holdout_scenes": len(holdout),
             "source_counts": dict(Counter(s["source"] for s in source)),
             "normalization_mean": mean.tolist(),
             "normalization_std": std.tolist()}
    return source, holdout, audit


def satellite_batch(scene: dict, rng: np.random.Generator):
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    def move(x: np.ndarray) -> torch.Tensor:
        x = np.rot90(x, k, axes=(-2, -1))
        if flip:
            x = np.flip(x, axis=-1)
        return torch.from_numpy(np.ascontiguousarray(x))
    return (move(scene["image"])[None], move(scene["wet"]),
            move(scene["observed"]))


def satellite_callback(scenes: list[dict], seed: int):
    rng = np.random.default_rng(seed+816_497)
    stats = {"updates": 0, "sources": Counter(), "first_20_losses": [],
             "last_20_losses": []}
    def step(model: JointMaskedCNN, optimizer, batch_number: int):
        if batch_number % SAT_INTERVAL:
            return
        scene = scenes[int(rng.integers(len(scenes)))]
        image, wet, observed = satellite_batch(scene, rng)
        logits = model.satellite_scores(image)[0]
        loss, _ = weak_loss(logits, wet, observed, scene["source"])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite UNOSAT auxiliary loss")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        stats["updates"] += 1
        stats["sources"][scene["source"]] += 1
        if stats["updates"] <= 20:
            stats["first_20_losses"].append(float(loss.detach()))
        stats["last_20_losses"].append(float(loss.detach()))
        stats["last_20_losses"] = stats["last_20_losses"][-20:]
    return step, stats


def satellite_holdout(model: JointMaskedCNN, scenes: list[dict]):
    rows = []
    model.eval()
    with torch.no_grad():
        for s in scenes:
            logits = model.satellite_scores(torch.from_numpy(s["image"][None]))[0].numpy()
            wet = s["wet"].ravel()
            crop = s["static"][STATIC_NAMES.index(
                "worldcover_crop_fraction_2021")].ravel()
            if wet.sum() <= 0:
                continue
            rows.append({"date": s["day"], "wet_mass": float(wet.sum()),
                         "model_hits": float(np.dot(selected(logits.ravel(), .1), wet)),
                         "crop_hits": float(np.dot(selected(crop, .1), wet))})
    mass = sum(r["wet_mass"] for r in rows)
    return {"dates": rows, "positive_mass": mass,
            "model_top10_capture": sum(r["model_hits"] for r in rows)/mass,
            "crop_top10_capture": sum(r["crop_hits"] for r in rows)/mass}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--reuse-report-controls", action="store_true")
    parser.add_argument("--shuffle-satellite-labels", action="store_true")
    parser.add_argument("--path-joint-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_grid()
    elevation, _, _, terrain_audit = terrain(data)
    sat_source, sat_holdout, sat_audit = load_satellite()
    if args.shuffle_satellite_labels:
        # Preserve each scene's wet and analyzed-dry mass, but sever its
        # alignment with terrain, roads and cover. This controls for extra
        # optimizer updates and generic mapped-water class balance.
        for j, scene in enumerate(sat_source):
            permutation = np.random.default_rng(714_113+j).permutation(80*80)
            scene["wet"] = scene["wet"].ravel()[permutation].reshape(80,80)
            scene["observed"] = scene["observed"].ravel()[permutation].reshape(80,80)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    saved = torch.load(PRETRAIN_MODEL, map_location="cpu", weights_only=True)
    controls = None
    if args.reuse_report_controls:
        suffix = "" if args.seed_offset == 0 else f"_seed_{args.seed_offset}"
        control_path = REPORT_ONLY_JSON.with_name(
            REPORT_ONLY_JSON.stem+suffix+".json")
        controls = {f["fold"]: f for f in json.loads(control_path.read_text())["folds"]}
    records, states = [], {}
    for fold_id, ids in enumerate(folds, 1):
        if args.fold is not None and fold_id != args.fold:
            continue
        source = ~np.isin(district, ids)
        static, norm = lidar_static(data, source, elevation)
        mask_self_check(data, static)
        pre = SpatialNovelCNN()
        pre.load_state_dict(saved["states"][fold_id]["synthetic"])
        record = {"fold": fold_id,
                  "heldout_district_codes": [data["districts"][j]["code"]
                                             for j in ids],
                  "normalization": norm, "arms": {}}
        states[fold_id] = {}
        print("fold", fold_id, flush=True)
        if controls is not None:
            prior = controls[fold_id]
            if (prior["heldout_district_codes"] != record["heldout_district_codes"]
                or prior["normalization"] != norm):
                raise ValueError("Reused report controls differ in held-out districts")
            for name, old in (("scratch_report_only", "scratch"),
                              ("path_report_only", "path_pretrained")):
                record["arms"][name] = {
                    **prior["arms"][old], "auxiliary_updates": 0,
                    "auxiliary_source_counts": {},
                    "auxiliary_first20_mean": None,
                    "auxiliary_last20_mean": None,
                    "pathum_thani_unosat": None,
                    "reused_from": str(control_path),
                }
        for name, initial, joint in (
            ("scratch_report_only", None, False),
            ("path_report_only", pre, False),
            ("scratch_joint", None, True),
            ("path_joint", pre, True),
        ):
            if args.path_joint_only and name != "path_joint":
                continue
            if controls is not None and not joint:
                continue
            callback, aux_stats = satellite_callback(
                sat_source, SEED+fold_id*1000+args.seed_offset)
            model, fit = train(data, source, static,
                               SEED+fold_id*1000+args.seed_offset,
                               ARM, pretrained=initial,
                               model_factory=JointMaskedCNN,
                               auxiliary_step=callback if joint else None)
            valid = score_days(data, source, static, model,
                               range(VALID_START, TEST_START))
            alpha = max(ALPHAS, key=lambda a: (
                valid[str(a)]["isolated_group_capture"],
                valid[str(a)]["isolated_cell_capture"], -a))
            test = score_days(data, ~source, static, model,
                              range(TEST_START, N_DAYS))
            sat_test = satellite_holdout(model, sat_holdout) if joint else None
            record["arms"][name] = {
                "fit": fit, "auxiliary_updates": aux_stats["updates"],
                "auxiliary_source_counts": dict(aux_stats["sources"]),
                "auxiliary_first20_mean": float(np.mean(aux_stats["first_20_losses"]))
                    if joint else None,
                "auxiliary_last20_mean": float(np.mean(aux_stats["last_20_losses"]))
                    if joint else None,
                "source_validation": valid, "selected_alpha": alpha,
                "heldout_q4": test, "pathum_thani_unosat": sat_test}
            states[fold_id][name] = {k: v.cpu() for k, v in model.state_dict().items()}
            chosen = test[str(alpha)]
            print("fold", fold_id, name, "alpha", alpha,
                  "cells", chosen["isolated_hits"],
                  "groups", chosen["group_hits"],
                  "sat", None if sat_test is None else round(
                      sat_test["model_top10_capture"], 4), flush=True)
        records.append(record)
    pooled = {}
    for name in records[0]["arms"]:
        pooled[name] = {str(a): pool([f["arms"][name]["heldout_q4"][str(a)]
                                      for f in records]) for a in ALPHAS}
        pooled[name]["source_selected"] = pool([
            f["arms"][name]["heldout_q4"][str(f["arms"][name]["selected_alpha"])]
            for f in records])
        if args.fold is None:
            b = pooled[name]["0.0"]
            if (b["isolated_positive"], b["isolated_groups"],
                b["isolated_hits"], b["group_hits"]) != (332, 158, 65., 44.):
                raise ValueError("Frozen strict isolated queue changed")
    result = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": "Do pre-Q4 satellite target types improve spatial transfer of path-pretrained, target-masked urban report ranking?",
        "seed_offset": args.seed_offset,
        "report_only_controls_reused": args.reuse_report_controls,
        "satellite_labels_shuffled": args.shuffle_satellite_labels,
        "target": "Q4 Traffy report-positive strict isolated 250m cells; UNOSAT mapped-water auxiliary positive/dry extent",
        "auxiliary": "2021/2024 UNOSAT only; source-specific static adapter and head, shared spatial steps; one update per two report updates",
        "auxiliary_data": sat_audit,
        "heldout_satellite": "Pathum Thani 2021 UNOSAT scenes; region excluded from satellite training but same storm dates occur in other provinces",
        "excluded_satellite_regions": sorted(SAT_HOLDOUT|SAT_EXCLUDE),
        "report_training": "2023-2024 source districts; same exact target-3x3 masked adjacent-history group trainer in all arms",
        "report_selection": "Jan-Sep 2025 source districts, group then cell capture, then smaller correction",
        "report_test": "Q4 2025 excluded districts; 10% daily budget with 20% strict isolated reserve",
        "synthetic_pretraining_checkpoint": str(PRETRAIN_MODEL),
        "terrain": terrain_audit,
        "folds": records, "pooled_heldout_q4": pooled,
        "limitations": ["UNOSAT 2021 mapped water is largely a different target from urban reports",
                        "Unreported report cells are unknown, not verified dry",
                        "D8 paths are terrain proxies, not observed drainage",
                        "Present-day LiDAR and static layers are not vintage-matched",
                        "Q4 2025 has been repeatedly inspected; this is exploratory",
                        "No verified urban wet/dry extent or rain-scenario validation"],
    }
    suffix = "" if args.seed_offset == 0 else f"_seed_{args.seed_offset}"
    if args.shuffle_satellite_labels:
        suffix += "_shuffled_satellite_labels"
    if args.path_joint_only and not args.shuffle_satellite_labels:
        suffix += "_path_joint_only"
    if args.fold is not None:
        suffix += f"_fold_{args.fold}"
    out_json = OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    out_model = OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    out_json.write_text(json.dumps(result, indent=2)+"\n")
    torch.save({"states": states}, out_model)
    print("pooled", {name: (r["source_selected"]["isolated_hits"],
                             r["source_selected"]["group_hits"])
                     for name, r in pooled.items()}, flush=True)
    print("saved", out_json, out_model, flush=True)


if __name__ == "__main__":
    main()
