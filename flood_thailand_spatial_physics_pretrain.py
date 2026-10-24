"""Test synthetic surface-flow pretraining across held-out Thai places.

This stage uses only shared open static layers. It does not use flood reports,
satellite water labels, BMA drainage features, or the report-history mask.
Conservation and passive hydraulic-head direction are hard model rules; all
rates in the teacher are heuristic, not measured flood hydraulics.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flood_citywide_conservative_flow import (
    ConservativeFlowCNN, physics_self_check, scenario, teacher_step,
)
from flood_gistda_multiregion_data import SIZE
from flood_open_weak_data import all_specs, load_static
from flood_multi_target_citywide_data import OUT


OUT_JSON = OUT / "thailand_spatial_physics_pretrain_heldout_places.json"
OUT_MODEL = OUT / "thailand_spatial_physics_pretrain_heldout_places.pt"
HELDOUT = ("bang_kapi_core", "nong_chok", "chiang_mai_city", "hat_yai_city")
SEED = 9173
STEPS = 900
COMMON = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
LOSS_WEIGHTS = {"surface": 1., "flow": .4, "infiltration": .2}


class LocalOnlyFlow(ConservativeFlowCNN):
    """Same hard conservation/direction rules, without spatial convolutions."""

    def __init__(self):
        super().__init__()
        self.steps = nn.ModuleList([])


def load_places():
    specs = all_specs()
    missing = set(HELDOUT)-set(specs)
    if missing:
        raise ValueError(f"Missing held-out places: {sorted(missing)}")
    places = {name: load_static(name) for name in specs}
    if any(x.shape != (13, SIZE, SIZE) or not np.isfinite(x).all()
           for x in places.values()):
        raise ValueError("Invalid common static patch")
    for name in HELDOUT:
        a = specs[name]["utm_bounds"]
        for other in set(specs)-set(HELDOUT):
            b = specs[other]["utm_bounds"]
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                raise ValueError(f"Held-out patch overlaps training: {name}, {other}")
    return places


def static_templates(places, source):
    # Absolute altitude is not a flow direction. Center it separately in each
    # patch, then fit remaining channel scales on source places only.
    common = {}
    for name, raw in places.items():
        x = raw[list(COMMON)].copy()
        x[0] -= np.median(x[0])
        common[name] = x
    train_values = np.concatenate([common[n].reshape(10, -1)
                                   for n in source], axis=1)
    mean = train_values.mean(axis=1)
    std = np.maximum(train_values.std(axis=1), .05)
    templates = {}
    for name, x in common.items():
        image = np.zeros((20, SIZE, SIZE), np.float32)
        image[2] = .5
        image[3:13] = np.clip((x-mean[:, None, None])/std[:, None, None], -5, 5)
        image[13] = 1
        image[19] = 1
        templates[name] = image
    return templates, {"mean": mean.tolist(), "std": std.tolist(),
                       "absolute_elevation_center": "patch median",
                       "channels": ["SRTM elevation", "relative elevation", "slope",
                                    "OSM canal length", "OSM canal distance",
                                    "OSM road length", "OSM road distance",
                                    "WorldCover built", "WorldCover crop",
                                    "WorldCover tree/grass"]}


def sample(raw, template, rng, rain_multiplier=1.):
    candidate = np.ones((SIZE, SIZE), bool)
    surface, rain, pipe, gate = scenario(rng, candidate,
                                         np.zeros((SIZE, SIZE), np.float32))
    rain *= rain_multiplier
    available = surface+rain
    elevation = raw[0].astype(np.float32)
    cover = np.stack((raw[7], raw[9])).astype(np.float32)
    roads = np.clip(raw[5]/7., 0, 1)
    drainage = np.zeros((3, SIZE, SIZE), np.float32)
    truth = teacher_step(surface, rain, pipe, gate, elevation, candidate,
                         cover, np.zeros_like(elevation), roads, drainage)
    image = template.copy()
    image[0] = np.log1p(1000*available)
    args = (torch.from_numpy(image[None]),
            torch.from_numpy(available[None, None]),
            torch.from_numpy(pipe[None, None]),
            torch.from_numpy(elevation[None, None]))
    return args, truth


def train_loss(pred, truth):
    # The teacher's positive surface cells are the relevant water domain.
    wet = torch.from_numpy((truth["surface"] > .001)[None, None])
    pieces = {}
    for key, weight in LOSS_WEIGHTS.items():
        actual = torch.from_numpy(truth[key][None, None] if key != "flow"
                                  else truth[key][None])
        select = wet.expand_as(pred[key])
        delta_mm = (pred[key]-actual)[select]*1000
        pieces[key] = weight*F.smooth_l1_loss(delta_mm,
                                               torch.zeros_like(delta_mm))
    return sum(pieces.values())


def evaluate(models, places, templates, names, multiplier=1., repeats=12):
    rows = []
    for name in names:
        rng = np.random.default_rng(SEED+sum(name.encode())+round(multiplier*1000))
        measures = {key: [] for key in (*models, "persistence")}
        for _ in range(repeats):
            args, truth = sample(places[name], templates[name], rng, multiplier)
            available = args[1][0, 0].numpy()
            wet = truth["surface"] > .001
            if not np.any(wet):
                raise ValueError(f"No wet cells in {name}")
            measures["persistence"].append({
                "surface_mae_mm": float(np.abs(available[wet]-truth["surface"][wet]).mean()*1000)})
            with torch.no_grad():
                for label, model in models.items():
                    pred = model(*args)
                    surface = pred["surface"][0, 0].numpy()
                    flow = pred["flow"][0].numpy()
                    infiltration = pred["infiltration"][0, 0].numpy()
                    total_in = float((available+args[2][0, 0].numpy()).sum())
                    total_out = sum(float(pred[k].sum()) for k in
                                    ("surface", "pipe", "infiltration", "outlet"))
                    measures[label].append({
                        "surface_mae_mm": float(np.abs(surface[wet]-truth["surface"][wet]).mean()*1000),
                        "flow_mae_mm": float(np.abs(flow[:, wet]-truth["flow"][:, wet]).mean()*1000),
                        "infiltration_mae_mm": float(np.abs(infiltration[wet]-truth["infiltration"][wet]).mean()*1000),
                        "balance_error_fraction": abs(total_in-total_out)/max(total_in, 1e-8)})
        rows.append({"place": name, "rain_multiplier": multiplier,
                     "wet_samples": repeats,
                     "models": {label: {metric: round(float(np.mean([v[metric] for v in values])), 6)
                                       for metric in values[0]}
                                for label, values in measures.items()}})
    return rows


def rollout_evaluate(models, places, templates, names, multiplier=1.,
                     repeats=12, horizon=4):
    """Reapply the learned one-step rule without teacher forcing."""
    rows = []
    for name in names:
        raw = places[name]
        rng = np.random.default_rng(SEED+sum(name.encode())+round(multiplier*1000))
        measures = {key: [] for key in (*models, "persistence")}
        candidate = np.ones((SIZE, SIZE), bool)
        elevation = raw[0].astype(np.float32)
        cover = np.stack((raw[7], raw[9])).astype(np.float32)
        roads = np.clip(raw[5]/7., 0, 1)
        drainage = np.zeros((3, SIZE, SIZE), np.float32)
        pipe = np.zeros((SIZE, SIZE), np.float32)
        gate = np.full((SIZE, SIZE), .5, np.float32)
        for _ in range(repeats):
            args, _ = sample(raw, templates[name], rng, multiplier)
            initial = args[1][0, 0].numpy()
            teacher = initial.copy()
            for step in range(horizon):
                truth = teacher_step(teacher, np.zeros_like(teacher), pipe,
                                     gate, elevation, candidate, cover,
                                     np.zeros_like(teacher), roads, drainage)
                teacher = truth["surface"]
            wet = teacher > .001
            measures["persistence"].append(float(np.abs(initial[wet]-teacher[wet]).mean()*1000))
            with torch.no_grad():
                for label, model in models.items():
                    surface = args[1]
                    pipe_tensor = args[2]
                    image = args[0].clone()
                    for step in range(horizon):
                        image[:, 0:1] = torch.log1p(1000*surface)
                        image[:, 1:2] = torch.log1p(1000*pipe_tensor)
                        pred = model(image, surface, pipe_tensor, args[3])
                        surface, pipe_tensor = pred["surface"], pred["pipe"]
                    got = surface[0, 0].numpy()
                    measures[label].append(float(np.abs(got[wet]-teacher[wet]).mean()*1000))
        rows.append({"place": name, "rain_multiplier": multiplier,
                     "horizon_steps": horizon,
                     "surface_mae_mm": {label: round(float(np.mean(values)), 6)
                                        for label, values in measures.items()}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=0)
    options = parser.parse_args()
    global SEED
    SEED += options.seed_offset
    torch.set_num_threads(4)
    physics_self_check()
    places = load_places()
    source = [n for n in places if n not in HELDOUT]
    templates, normalization = static_templates(places, source)
    models = {}
    histories = {}
    for label, factory in (("spatial", ConservativeFlowCNN),
                           ("local_only", LocalOnlyFlow)):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = factory()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002,
                                      weight_decay=.02)
        loss_history = []
        for step in range(STEPS):
            name = source[int(rng.integers(len(source)))]
            args, truth = sample(places[name], templates[name], rng)
            pred = model(*args)
            loss = train_loss(pred, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach()))
            if (step+1) % 150 == 0:
                print(label, "step", step+1, "loss", round(float(np.mean(loss_history[-150:])), 5), flush=True)
        model.eval()
        models[label] = model
        histories[label] = {"first_100_loss": round(float(np.mean(loss_history[:100])), 5),
                            "last_100_loss": round(float(np.mean(loss_history[-100:])), 5)}
    torch.manual_seed(SEED)
    models["untrained_spatial"] = ConservativeFlowCNN().eval()
    heldout = evaluate(models, places, templates, HELDOUT)
    doubled = evaluate(models, places, templates, HELDOUT, multiplier=2.)
    rollout = rollout_evaluate(models, places, templates, HELDOUT)
    rollout_doubled = rollout_evaluate(models, places, templates, HELDOUT,
                                      multiplier=2.)
    source_eval = evaluate(models, places, templates, source[:4])
    result = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Learn surface-flow response to synthetic storms in unseen Thai places before any masked report training",
        "target_status": "Heuristic synthetic water states, not observed inundation or verified flood extent",
        "sources": "24 cached 250m Thai patches: SRTM, OSM roads/canals, WorldCover 2021",
        "source_places": source, "heldout_places": list(HELDOUT),
        "seed_offset": options.seed_offset,
        "normalization": normalization, "steps_per_model": STEPS,
        "loss_weights": LOSS_WEIGHTS, "training": histories,
        "heldout_normal_rain": heldout, "heldout_double_rain": doubled,
        "heldout_four_step_rollout": rollout,
        "heldout_four_step_double_rain": rollout_doubled,
        "source_sample_normal_rain": source_eval,
        "constraints": ["Exact nonnegative water continuity over surface, pipe, infiltration and outlet ledger",
                        "Four-neighbor surface transfer only toward lower hydraulic head"],
        "limits": ["Teacher infiltration and mobility rates are invented; successful imitation does not validate flood prediction",
                   "One-step training; four-step rollout is an uncalibrated synthetic stress test, not a measured hydrograph",
                   "SRTM at 250m misses street-scale drainage, curbs and buildings",
                   "Shared Thai layers exclude Bangkok-only BMA pipes, pumps, gates and LiDAR",
                   "Held-out city patches do not certify transfer to a report-site mask task"]}
    suffix = f"_seed_{options.seed_offset}" if options.seed_offset else ""
    out_json = OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    out_model = OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    torch.save({"spatial": models["spatial"].state_dict(),
                "local_only": models["local_only"].state_dict(),
                "normalization": normalization}, out_model)
    print("heldout", [(r["place"], r["models"]["spatial"]["surface_mae_mm"],
                       r["models"]["local_only"]["surface_mae_mm"])
                      for r in heldout], flush=True)


if __name__ == "__main__":
    main()
