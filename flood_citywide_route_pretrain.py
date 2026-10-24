"""Synthetic terrain-routing pretraining for the strict isolated-site CNN.

The auxiliary target is NOT observed flooding. A sparse rain/runoff source is
routed down a 250 m D8 graph made from cached BMA LiDAR, for at most five hops.
Only the shared spatial trunk and static-feature entry weights transfer to the
Traffy report ranker. Each fold fits on source districts, then scores Q4 2025
in excluded districts using the frozen 10%/20%-reserve queue.

Run: .venv/bin/python flood_citywide_route_pretrain.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.stats import spearmanr
from torch.nn import functional as F

from flood_bangkok_transfer import CACHE as BMA_CACHE
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_isolated_site_ranker import day_inputs, isolated, queue, samples
from flood_citywide_spatial_novel_queue import (
    STATIC_NAMES, SpatialNovelCNN, top_weights,
)
from flood_citywide_uphill_paved import flow_edges, mosaic_bma_dtm
from flood_multi_target_citywide_data import OUT
from flood_open_weak_data import FEATURE_NAMES


OUT_JSON = OUT / "q4_2025_route_pretrain_isolated_fivefold.json"
OUT_MODEL = OUT / "q4_2025_route_pretrain_isolated_fivefold.pt"
TERRAIN_CACHE = BMA_CACHE / "citywide_route_pretrain_lidar_250m.npz"
STATIC_COLUMNS = [FEATURE_NAMES.index(n) for n in STATIC_NAMES]
HOPS = 5
DECAY = .72
EPOCHS = 2


def terrain(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Cache the citywide DTM and D8 graph, with strict grid alignment checks."""
    flat = data["candidate_flat"]
    if TERRAIN_CACHE.exists():
        with np.load(TERRAIN_CACHE) as saved:
            if not np.array_equal(saved["candidate_flat"], flat):
                raise ValueError("Cached terrain differs from report grid")
            return (saved["elevation"], saved["node_flat"], saved["down"],
                    json.loads(str(saved["audit"].item())))
    z, dem_audit = mosaic_bma_dtm(data["districts"], data["candidate"])
    (node_flat, down, _), edge_audit = flow_edges(z, data["candidate"])
    if len(node_flat) < .98*len(flat):
        raise ValueError("Too little valid DTM coverage")
    audit = {"dem": dem_audit, "edges": edge_audit,
             "valid_node_fraction": round(len(node_flat)/len(flat), 5)}
    TERRAIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TERRAIN_CACHE, elevation=z, node_flat=node_flat,
                        down=down, candidate_flat=flat, audit=json.dumps(audit))
    return z, node_flat, down, audit


def lidar_static(data: dict, source: np.ndarray, elevation: np.ndarray):
    """Replace three SRTM channels with source-normalized LiDAR terrain."""
    candidate = data["candidate"]
    good = candidate & np.isfinite(elevation)
    if good.sum() < .98*candidate.sum():
        raise ValueError("LiDAR coverage below 98%")
    nearest = distance_transform_edt(~good, return_distances=False,
                                     return_indices=True)
    z = elevation[tuple(nearest)].astype(np.float32)
    local_mean = gaussian_filter(z*good, 5, mode="nearest") / np.maximum(
        gaussian_filter(good.astype(np.float32), 5, mode="nearest"), 1e-4)
    dy, dx = np.gradient(z, 250.)
    lidar = np.stack((z, z-local_mean, np.hypot(dx, dy)))[:, candidate].T
    raw = data["static"][:, STATIC_COLUMNS].astype(np.float32).copy()
    raw[:, :3] = lidar
    mean = raw[source].mean(axis=0)
    std = np.maximum(raw[source].std(axis=0), .05)
    normed = np.clip((raw-mean)/std, -5, 5)
    image = np.zeros((len(STATIC_NAMES), *SHAPE), np.float32)
    image[:, candidate] = normed.T
    return image, {"names": ["lidar_elevation_m", "lidar_relative_1250m_m",
                             "lidar_slope", *STATIC_NAMES[3:]],
                   "mean": mean.tolist(), "std": std.tolist()}


def route(source: np.ndarray, node_flat: np.ndarray,
          down: np.ndarray) -> np.ndarray:
    """Accumulated source arriving from upstream, excluding the source cell."""
    current = source.ravel()[node_flat].astype(np.float32).copy()
    accumulated = np.zeros_like(current)
    valid = down >= 0
    for _ in range(HOPS):
        following = np.zeros_like(current)
        np.add.at(following, down[valid], DECAY*current[valid])
        accumulated += following
        current = following
    out = np.zeros(SHAPE, np.float32)
    out.ravel()[node_flat] = np.log1p(accumulated)
    return out


def route_self_check() -> None:
    toy = np.zeros(SHAPE, np.float32)
    toy[0, 0] = 1.
    got = route(toy, np.asarray([0, 1, 2]),
                np.asarray([1, 2, -1]))
    expected = np.log1p([0., DECAY, DECAY**2])
    if (not np.allclose(got[0, :3], expected) or
        np.count_nonzero(got) != 2):
        raise ValueError("Synthetic route direction or local-source exclusion changed")


def scenario(rng: np.random.Generator, data: dict, static: np.ndarray,
             scope_grid: np.ndarray, built: np.ndarray, node_flat: np.ndarray,
             down: np.ndarray):
    """Vary storm footprints and runoff source; do not use report outcomes."""
    h, w = SHAPE
    noise = rng.normal(size=SHAPE).astype(np.float32)
    rain = gaussian_filter(noise, rng.uniform(3, 12), mode="reflect")
    rain = (rain-rain.min())/max(float(rain.max()-rain.min()), 1e-6)
    # Variable local pulses keep topology, rather than fixed terrain, useful.
    pulse = (rng.random(SHAPE) < rng.uniform(.2, .6)).astype(np.float32)
    runoff = ((.15 + .75*built) * (.15 + 1.5*rain) * pulse *
              scope_grid).astype(np.float32)
    image = np.concatenate((np.zeros((4, h, w), np.float32), static,
                            data["candidate"][None].astype(np.float32),
                            np.ones((1, h, w), np.float32)))
    image[0] = runoff
    target = route(runoff, node_flat, down)
    return image, target


def pretrain(data: dict, source: np.ndarray, static: np.ndarray,
             node_flat: np.ndarray, down: np.ndarray, seed: int,
             steps: int, batch_size: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 491)
    model = SpatialNovelCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    scope = np.zeros(SHAPE, bool)
    scope.ravel()[data["candidate_flat"]] = source
    built = np.zeros(SHAPE, np.float32)
    built[data["candidate"]] = np.clip(data["static"][:, 7], 0, 1)
    source_flat = data["candidate_flat"][source]
    losses = []
    model.train()
    for step in range(steps):
        image, target = scenario(rng, data, static, scope, built,
                                 node_flat, down)
        values = target.ravel()[source_flat]
        active = np.flatnonzero(values > .02)
        inactive = np.flatnonzero(values <= .02)
        if not len(active) or not len(inactive):
            raise ValueError("Synthetic scenario has no active/inactive sites")
        indices = np.r_[rng.choice(active, batch_size//2, replace=True),
                        rng.choice(inactive, batch_size-batch_size//2,
                                   replace=True)]
        batch = patches(image, source_flat, indices)
        truth = torch.from_numpy(values[indices].copy())
        prediction = model.patch_scores(batch)
        loss = F.smooth_l1_loss(prediction, truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 40 == 0:
            print("route step", step+1, "loss", round(np.mean(losses[-40:]), 5),
                  flush=True)
    model.eval()
    heldout = ~source
    scope[:] = False
    scope.ravel()[data["candidate_flat"]] = heldout
    heldout_flat = data["candidate_flat"][heldout]
    pred, truth = [], []
    with torch.no_grad():
        for _ in range(3):
            image, target = scenario(rng, data, static, scope, built,
                                     node_flat, down)
            choose = rng.choice(len(heldout_flat),
                                min(2000, len(heldout_flat)), replace=False)
            pred.append(model.patch_scores(patches(
                image, heldout_flat, choose)).numpy())
            truth.append(target.ravel()[heldout_flat[choose]])
    pred, truth = np.concatenate(pred), np.concatenate(truth)
    quality = {"steps": steps, "loss_first_20": round(np.mean(losses[:20]), 5),
               "loss_last_20": round(np.mean(losses[-20:]), 5),
               "heldout_synthetic_spearman": round(float(spearmanr(
                   pred, truth).statistic), 5),
               "heldout_synthetic_mae": round(float(np.mean(np.abs(pred-truth))), 5)}
    return model, quality


def initialize_report_model(seed: int, pretrained: SpatialNovelCNN | None):
    torch.manual_seed(seed)
    model = SpatialNovelCNN()
    if pretrained is not None:
        with torch.no_grad():
            model.entry.weight[:, 4:].copy_(pretrained.entry.weight[:, 4:])
            model.entry.weight[:, :4].zero_()  # synthetic source is not history
            model.entry.bias.copy_(pretrained.entry.bias)
            model.steps.load_state_dict(pretrained.steps.state_dict())
            # Synthetic runoff head is discarded; report head starts fresh.
    return model


def finetune(data: dict, source: np.ndarray, static: np.ndarray,
             seed: int, pretrained: SpatialNovelCNN | None):
    rng = np.random.default_rng(seed)
    scope = np.zeros(SHAPE, bool)
    scope.ravel()[data["candidate_flat"]] = source
    days = [d for d in range(TRAIN_START, VALID_START)
            if np.any((data["reports"][d] > 0) & isolated(data, d) & scope)]
    model = initialize_report_model(seed, pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    losses, n_groups = [], 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for day in rng.permutation(days):
            image, _, simple, _ = day_inputs(data, int(day), static)
            picked = samples(data, int(day), scope, simple, rng, limit=12)
            if picked is None:
                continue
            positive, comparison, groups = picked
            n_groups += groups
            batch = patches(image, data["candidate_flat"],
                            np.r_[positive, comparison])
            if torch.any(batch[:, :4, RADIUS-1:RADIUS+2,
                                      RADIUS-1:RADIUS+2] != 0):
                raise ValueError("Isolated report history exposed")
            score = model.patch_scores(batch)
            loss = F.softplus(score[len(positive):]-score[:len(positive)]).mean()
            loss = loss + .0005*score.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(round(float(np.mean(epoch_losses)), 5))
        print("report", "pretrain" if pretrained else "scratch", "epoch",
              epoch+1, "loss", losses[-1], flush=True)
    model.eval()
    return model, {"train_days": len(days), "groups_across_epochs": n_groups,
                   "epoch_loss": losses}


def score_days(data: dict, scope: np.ndarray, static: np.ndarray,
               models: dict, days: range):
    names = ("history_no_reserve", "distant_history", *models)
    rows = {name: {"all_positive": 0, "all_hits": 0.,
                   "isolated_positive": 0, "isolated_hits": 0.,
                   "isolated_groups": 0, "group_hits": 0.}
            for name in names}
    chosen = np.flatnonzero(scope)
    grid_flat = data["candidate_flat"][chosen]
    for day in days:
        image, base, simple, _ = day_inputs(data, day, static)
        eligible = isolated(data, day)[data["candidate"]][chosen]
        observed = (data["reports"][day].ravel()[data["candidate_flat"]]
                    > 0)[chosen]
        isolated_positive = observed & eligible
        group_grid = np.zeros(SHAPE, bool)
        group_grid.ravel()[grid_flat] = isolated_positive
        groups, count = label(group_grid, structure=np.ones((3, 3), bool))
        group_ids = groups.ravel()[grid_flat]
        weights = {
            "history_no_reserve": top_weights(base[chosen], .1*len(chosen)),
            "distant_history": queue(base[chosen], simple[chosen], eligible),
        }
        for name, model in models.items():
            scores = model.full_scores(image, data["candidate"])
            weights[name] = queue(base[chosen], scores[chosen], eligible)
        for name, w in weights.items():
            r = rows[name]
            r["all_positive"] += int(observed.sum())
            r["all_hits"] += float(w[observed].sum())
            r["isolated_positive"] += int(isolated_positive.sum())
            r["isolated_hits"] += float(w[isolated_positive].sum())
            r["isolated_groups"] += count
            r["group_hits"] += sum(float(w[group_ids == j].max())
                                    for j in range(1, count+1))
    for r in rows.values():
        r["all_capture"] = round(r["all_hits"]/r["all_positive"], 5)
        r["isolated_cell_capture"] = round(
            r["isolated_hits"]/r["isolated_positive"], 5)
        r["isolated_group_capture"] = round(
            r["group_hits"]/r["isolated_groups"], 5)
    return rows


def pooled(records: list[dict], field: str):
    result = {}
    keys = ("all_positive", "all_hits", "isolated_positive",
            "isolated_hits", "isolated_groups", "group_hits")
    for name in records[0][field]:
        row = {key: sum(f[field][name][key] for f in records) for key in keys}
        row["all_capture"] = round(row["all_hits"]/row["all_positive"], 5)
        row["isolated_cell_capture"] = round(
            row["isolated_hits"]/row["isolated_positive"], 5)
        row["isolated_group_capture"] = round(
            row["group_hits"]/row["isolated_groups"], 5)
        result[name] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-steps", type=int, default=100)
    parser.add_argument("--pretrain-batch", type=int, default=192)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    if args.pretrain_steps < 20 or args.pretrain_batch < 16:
        raise ValueError("Too few synthetic training examples")
    route_self_check()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_grid()
    elevation, node_flat, down, terrain_audit = terrain(data)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    records, states = [], {}
    for fold_id, ids in enumerate(folds, 1):
        source = ~np.isin(district, ids)
        static, norm = lidar_static(data, source, elevation)
        seed = SEED + fold_id*1000 + args.seed_offset
        print("fold", fold_id, "source", int(source.sum()), flush=True)
        pre, pre_fit = pretrain(data, source, static, node_flat, down,
                                seed, args.pretrain_steps, args.pretrain_batch)
        scratch, scratch_fit = finetune(data, source, static, seed, None)
        tuned, tuned_fit = finetune(data, source, static, seed, pre)
        models = {"scratch_lidar_cnn": scratch,
                  "route_pretrained_cnn": tuned}
        validation = score_days(data, source, static, models,
                                range(VALID_START, TEST_START))
        test = score_days(data, ~source, static, models,
                          range(TEST_START, N_DAYS))
        records.append({"fold": fold_id,
                        "heldout_district_codes": [data["districts"][j]["code"]
                                                   for j in ids],
                        "normalization": norm, "synthetic_fit": pre_fit,
                        "scratch_fit": scratch_fit,
                        "pretrained_fit": tuned_fit,
                        "source_validation": validation, "heldout_q4": test})
        states[fold_id] = {"scratch": {k: v.cpu() for k, v in
                                      scratch.state_dict().items()},
                           "pretrained": {k: v.cpu() for k, v in
                                         tuned.state_dict().items()}}
        print("fold", fold_id, {name: (r["isolated_hits"],
                                     r["group_hits"]) for name, r in
                                     test.items()}, flush=True)
    result = {"built_utc": datetime.now(timezone.utc).isoformat(
                  timespec="seconds"),
              "question": "Does synthetic terrain-route pretraining improve geographically held-out isolated report-site ranking?",
              "target": "Traffy report-positive 250 m cell-days; NOT verified flooding",
              "synthetic_target": "Log1p runoff source arriving through 1-5 D8 LiDAR downhill hops, excluding local source",
              "scenario": "Spatially varied synthetic rain/runoff pulses weighted by WorldCover built fraction; no report labels in pretraining",
              "model": "SpatialNovelCNN; transfer static-entry weights and spatial trunk, reset synthetic-source entry weights and prediction head",
              "terrain": terrain_audit, "pretrain_steps": args.pretrain_steps,
              "pretrain_batch": args.pretrain_batch,
              "seed_offset": args.seed_offset,
              "training_dates": ["2023-01-01", "2024-12-31"],
              "source_validation_dates": ["2025-01-01", "2025-09-30"],
              "geographic_test_dates": ["2025-10-01", "2025-12-31"],
              "queue": "10% of heldout fold cells per day; 20% of slots reserved for no own or adjacent prior-year report",
              "folds": records,
              "pooled_source_validation": pooled(records, "source_validation"),
              "pooled_heldout_q4": pooled(records, "heldout_q4"),
              "limitations": ["D8 route targets are synthetic and ignore drains, pumps, gates, barriers and tide",
                              "Cached BMA LiDAR is present-day terrain, not a vintage-matched 2023-2025 archive",
                              "Unreported comparison cells are unknown, not verified dry",
                              "Q4 2025 has been inspected repeatedly; these results are exploratory",
                              "No rain-conditioned flood probability or validated scenario prediction"]}
    if (result["pooled_heldout_q4"]["distant_history"]["isolated_positive"] != 332 or
        result["pooled_heldout_q4"]["distant_history"]["isolated_hits"] != 65.):
        raise ValueError("Frozen isolated queue changed")
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"states": states}, OUT_MODEL)
    print("pooled", {k: (v["isolated_hits"], v["group_hits"],
                         v["all_capture"]) for k, v in
                         result["pooled_heldout_q4"].items()}, flush=True)
    print("saved", OUT_JSON, OUT_MODEL, flush=True)


if __name__ == "__main__":
    main()
