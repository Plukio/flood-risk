"""Test transferable terrain paths carrying a WorldCover built-area rain proxy.

The target is Bangkok Traffy report-positive 250 m cell-days, not verified
inundation. Full local-day Open-Meteo rain makes this a retrospective test.
Run: .venv/bin/python flood_citywide_uphill_paved.py
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix
from scipy.special import expit

from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, capture, load_grid,
)
from flood_multi_target_citywide_data import OUT, X0, Y0, TILE_M, COLS, ROWS
from flood_multi_target_citywide_eval import paired_interval
from flood_bangkok_transfer import CACHE as BMA_CACHE
from flood_bangkok_spatial_cnn_citywide import HOLDOUT
from flood_open_weak_bangkok_holdout_eval import selected


GRID_M = 250
MIN_DROP_M = 0.05
MAX_DROP_M = 5.0
MIN_ROUTE_M = 500.0
MAX_ROUTE_M = 3000.0
DECAY_M = 1500.0
PAIR_PER_POSITIVE = 8
RIDGE = 0.02
SEED = 7701
CACHE_FILE = BMA_CACHE / "citywide_uphill_built_250m_v1.npz"
OUT_FILE = OUT / "q4_2025_uphill_built_rain_ranker_monotone.json"
KINDS = ("local", "distance", "reverse", "uphill")
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def mosaic_bma_dtm(districts: list[dict], candidate: np.ndarray) -> tuple[np.ndarray, dict]:
    """Average cached BMA DTM samples into one grid, masking district overlaps."""
    transform = from_origin(X0, Y0 + ROWS * TILE_M, GRID_M, GRID_M)
    numerator = np.zeros(SHAPE, np.float64)
    denominator = np.zeros(SHAPE, np.float64)
    source_resolution = []
    missing_sources = []
    for district in districts:
        path = BMA_CACHE / f"terrain_{district['code']}.tif"
        if not path.exists():
            missing_sources.append(district["code"])
            continue
        with rasterio.open(path) as src:
            if src.crs.to_epsg() != 32647:
                raise ValueError(f"Unexpected DTM CRS: {path}")
            value = src.read(1).astype(np.float32)
            inside = geometry_mask([district["polygon"].__geo_interface__],
                                   src.shape, src.transform, invert=True)
            good = inside & np.isfinite(value) & (value > -20) & (value < 60)
            value[~good] = -9999
            mean = np.full(SHAPE, -9999.0, np.float32)
            fraction = np.zeros(SHAPE, np.float32)
            reproject(value, mean, src_transform=src.transform, src_crs=src.crs,
                      src_nodata=-9999, dst_transform=transform,
                      dst_crs="EPSG:32647", dst_nodata=-9999,
                      resampling=Resampling.average)
            reproject(good.astype(np.float32), fraction,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:32647",
                      resampling=Resampling.average)
            usable = (mean > -100) & (fraction > 0)
            numerator[usable] += mean[usable] * fraction[usable]
            denominator[usable] += fraction[usable]
            source_resolution.append(float(src.res[0]))
    if missing_sources:
        raise FileNotFoundError(f"Missing cached BMA DTM for {missing_sources}")
    result = np.divide(numerator, denominator,
                       out=np.full(SHAPE, np.nan), where=denominator > 0)
    usable = candidate & np.isfinite(result)
    if usable.sum() < candidate.sum() * .98:
        raise ValueError(f"Only {usable.sum()}/{candidate.sum()} candidate DTM cells")
    audit = {"source": "cached BMA DTM_LiDAR exportImage at 512x512 per district",
             "districts": len(source_resolution),
             "source_pixel_median_m": round(float(np.median(source_resolution)), 2),
             "source_pixel_range_m": [round(min(source_resolution), 2),
                                      round(max(source_resolution), 2)],
             "candidate_terrain_cells": int(usable.sum()),
             "candidate_missing_terrain_cells": int((candidate & ~usable).sum()),
             "candidate_median_elevation_m": round(float(np.median(result[usable])), 3)}
    return result.astype(np.float32), audit


def flow_edges(elevation: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, dict]:
    """Create D8 downhill links without using report labels or district IDs."""
    flat = np.flatnonzero(candidate.ravel() & np.isfinite(elevation.ravel()))
    lookup = np.full(candidate.size, -1, np.int32)
    lookup[flat] = np.arange(len(flat), dtype=np.int32)
    h, w = candidate.shape
    down = np.full(len(flat), -1, np.int32)
    length = np.zeros(len(flat), np.float32)
    drops = []
    for node, index in enumerate(flat):
        row, col = divmod(int(index), w)
        z = float(elevation[row, col])
        best_slope = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                neighbor = int(lookup[nr * w + nc])
                if neighbor < 0:
                    continue
                distance = GRID_M * math.hypot(dr, dc)
                drop = z - float(elevation[nr, nc])
                if not (MIN_DROP_M <= drop <= MAX_DROP_M):
                    continue
                slope = drop / distance
                if slope > best_slope:
                    best_slope, down[node], length[node] = slope, neighbor, distance
        if down[node] >= 0:
            drops.append(z - float(elevation.ravel()[flat[down[node]]]))
    audit = {"valid_nodes": len(flat), "downhill_edges": int((down >= 0).sum()),
             "sinks_or_boundary_nodes": int((down < 0).sum()),
             "drop_median_m": round(float(np.median(drops)), 3),
             "drop_p99_m": round(float(np.percentile(drops, 99)), 3),
             "minimum_drop_m": MIN_DROP_M, "maximum_drop_m": MAX_DROP_M}
    return (flat, down, length), audit


def path_matrix(down: np.ndarray, length: np.ndarray) -> tuple[csr_matrix, dict]:
    """Source j sends a decaying message to downstream i at 0.5–3 km path length."""
    rows, cols, weights = [], [], []
    source_reaches = np.zeros(len(down), np.int16)
    for source in range(len(down)):
        current, traveled = source, 0.0
        for _ in range(16):
            target = int(down[current])
            if target < 0:
                break
            traveled += float(length[current])
            if traveled > MAX_ROUTE_M:
                break
            if traveled >= MIN_ROUTE_M:
                rows.append(target)
                cols.append(source)
                weights.append(math.exp(-traveled / DECAY_M))
                source_reaches[source] += 1
            current = target
    matrix = csr_matrix((np.asarray(weights, np.float32),
                         (np.asarray(rows, np.int32), np.asarray(cols, np.int32))),
                        shape=(len(down), len(down)))
    return matrix, {"source_nodes_reaching_500m": int((source_reaches > 0).sum()),
                    "source_nodes_reaching_500m_fraction": round(float((source_reaches > 0).mean()), 4),
                    "path_message_pairs": int(matrix.nnz),
                    "minimum_path_m": MIN_ROUTE_M, "maximum_path_m": MAX_ROUTE_M,
                    "decay_m": DECAY_M}


def distance_kernel() -> np.ndarray:
    radius = math.ceil(MAX_ROUTE_M / GRID_M)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    distance = GRID_M * np.hypot(yy, xx)
    kernel = np.where((distance >= MIN_ROUTE_M) & (distance <= MAX_ROUTE_M),
                      np.exp(-distance / DECAY_M), 0.0).astype(np.float32)
    return kernel


def build_inputs(data: dict) -> tuple[dict, dict]:
    candidate = data["candidate"]
    candidate_flat = data["candidate_flat"]
    built = np.clip(data["static"][:, 7], 0, 1).astype(np.float32)
    if CACHE_FILE.exists():
        with np.load(CACHE_FILE) as saved:
            if not np.array_equal(saved["candidate_flat"], candidate_flat):
                raise ValueError("Cached terrain path grid differs from report grid")
            audit = json.loads(str(saved["audit"].item()))
            if not np.allclose(saved["built"], built):
                raise ValueError("Cached WorldCover built fraction changed")
            if "cross_district_edges" in audit["edge"]:
                return {key: saved[key] for key in ("uphill", "distance", "reverse", "built")}, audit
    elevation, dem_audit = mosaic_bma_dtm(data["districts"], candidate)
    (flat, down, length), edge_audit = flow_edges(elevation, candidate)
    matrix, route_audit = path_matrix(down, length)
    node_to_candidate = np.searchsorted(candidate_flat, flat)
    if not np.array_equal(candidate_flat[node_to_candidate], flat):
        raise ValueError("Terrain nodes do not align to candidate cells")
    district = data["district_id"].ravel()[candidate_flat][node_to_candidate]
    linked = down >= 0
    edge_audit["cross_district_edges"] = int(np.count_nonzero(
        district[linked] != district[down[linked]]))
    source = np.zeros((len(flat), 9), np.float32)
    source_tile = data["tile_id"][node_to_candidate]
    source[np.arange(len(flat)), source_tile] = built[node_to_candidate] * (GRID_M ** 2 / 1e6)
    uphill_valid = matrix @ source
    reverse_valid = matrix.T @ source
    uphill = np.zeros((len(candidate_flat), 9), np.float32)
    reverse = np.zeros_like(uphill)
    uphill[node_to_candidate] = uphill_valid
    reverse[node_to_candidate] = reverse_valid
    kernel = distance_kernel()
    distance = np.zeros_like(uphill)
    scale = float(matrix.sum() / (len(flat) * kernel.sum()))
    for tile in range(9):
        source_grid = np.zeros(SHAPE, np.float32)
        source_grid.ravel()[candidate_flat[data["tile_id"] == tile]] = (
            built[data["tile_id"] == tile] * (GRID_M ** 2 / 1e6))
        distance[:, tile] = fftconvolve(source_grid, kernel, mode="same")[candidate] * scale
    audit = {"dem": dem_audit, "edge": edge_audit, "route": route_audit,
             "built_proxy": "ESA WorldCover 2021 built fraction; includes buildings and mapped built surfaces, not measured pavement/permeability",
             "distance_control": "same 500–3000 m radial reach, exponential decay and globally matched total weight",
             "distance_weight_scale": round(scale, 6),
             "uphill_destination_cells": int((uphill.sum(axis=1) > 0).sum()),
             "uphill_destination_fraction": round(float((uphill.sum(axis=1) > 0).mean()), 4),
             "uphill_median_nonzero_built_km2": round(float(np.median(
                 uphill.sum(axis=1)[uphill.sum(axis=1) > 0])), 5)}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, uphill=uphill, distance=distance,
                        reverse=reverse, built=built, candidate_flat=candidate_flat,
                        audit=json.dumps(audit))
    return {"uphill": uphill, "distance": distance,
            "reverse": reverse, "built": built}, audit


def base_score(data: dict, day: int) -> np.ndarray:
    counts365 = (data["prefix"][day] - data["prefix"][day - 365]).astype(np.float32)
    counts90 = (data["prefix"][day] - data["prefix"][day - 90]).astype(np.float32)
    near = gaussian_filter(counts365, 1, mode="constant")
    mask = data["candidate"]
    return (np.log1p(counts365[mask]) + .4 * np.log1p(counts90[mask]) +
            .4 * np.log1p(near[mask])).astype(np.float32)


def response_features(data: dict, inputs: dict, kind: str, day: int) -> np.ndarray:
    rain = np.expm1(data["rain_by_tile"][day].astype(np.float32))
    local = np.log1p(inputs["built"] * rain[data["tile_id"], 0])
    if kind == "local":
        return local[:, None]
    by_tile = inputs[kind]
    static = np.log1p(by_tile.sum(axis=1))
    today = np.log1p(by_tile @ rain[:, 0])
    peak = np.log1p(by_tile @ rain[:, 1])
    return np.column_stack((local, static, today, peak)).astype(np.float32)


def training_pairs(data: dict, train_mask: np.ndarray | None = None) -> tuple[list, dict]:
    rng = np.random.default_rng(SEED)
    candidate_flat = data["candidate_flat"]
    if train_mask is None:
        train_mask = np.ones(len(candidate_flat), bool)
    pairs = []
    audit = {"positive_cell_days": 0, "pair_count": 0, "days_with_positives": 0}
    for day in range(TRAIN_START, VALID_START):
        truth = data["reports"][day].ravel()[candidate_flat]
        positive = np.flatnonzero(truth.astype(bool) & train_mask)
        if not len(positive):
            continue
        base = base_score(data, day)
        unlabeled = np.flatnonzero(~truth.astype(bool) & train_mask)
        hot_size = max(1000, int(.1 * len(base)))
        hot = np.argpartition(base[unlabeled], -hot_size)[-hot_size:]
        hot = unlabeled[hot]
        pos = np.repeat(positive, PAIR_PER_POSITIVE)
        neg = np.empty(len(pos), np.int32)
        neg[0::2] = rng.choice(unlabeled, size=len(neg[0::2]), replace=True)
        neg[1::2] = rng.choice(hot, size=len(neg[1::2]), replace=True)
        pairs.append((day, pos.astype(np.int32), neg,
                      (base[pos] - base[neg]).astype(np.float32)))
        audit["positive_cell_days"] += len(positive)
        audit["pair_count"] += len(pos)
        audit["days_with_positives"] += 1
    return pairs, audit


def fit_response(data: dict, inputs: dict, pairs: list, kind: str) -> dict:
    deltas = []
    offsets = []
    for day, pos, neg, base_difference in pairs:
        features = response_features(data, inputs, kind, day)
        deltas.append(features[pos] - features[neg])
        offsets.append(base_difference)
    delta = np.concatenate(deltas).astype(np.float64)
    offset = np.concatenate(offsets).astype(np.float64)
    scale = np.maximum(delta.std(axis=0), 0.01)
    z = delta / scale

    def objective(beta):
        margin = offset + z @ beta
        loss = np.logaddexp(0, -margin).mean() + .5 * RIDGE * beta @ beta
        gradient = -(z.T @ expit(-margin)) / len(margin) + RIDGE * beta
        return loss, gradient

    # More rain cannot lower this learned residual when the other inputs are
    # fixed. The static path coefficient remains free to reveal a bad proxy.
    bounds = ([(0, None)] if kind == "local" else
              [(0, None), (None, None), (0, None), (0, None)])
    result = minimize(objective, np.zeros(z.shape[1]), method="L-BFGS-B",
                      jac=True, bounds=bounds, options={"maxiter": 100})
    if not result.success:
        raise RuntimeError(f"Response fit failed for {kind}: {result.message}")
    return {"kind": kind, "coefficient": (result.x / scale).tolist(),
            "standardized_coefficient": result.x.tolist(),
            "difference_scale": scale.tolist(), "training_loss": float(result.fun),
            "feature_names": (["local_built_x_full_day_rain"] if kind == "local" else
                              ["local_built_x_full_day_rain", "path_built_km2_log",
                               "path_built_x_full_day_rain_log",
                               "path_built_x_peak_3h_rain_log"])}


def score_period(data: dict, inputs: dict, models: dict, days: range,
                 alphas: dict | None = None,
                 geo_mask: np.ndarray | None = None) -> tuple[dict, np.ndarray, dict]:
    alphas = alphas or {kind: ALPHAS for kind in KINDS}
    candidate_flat = data["candidate_flat"]
    masks = {"citywide": np.ones(len(candidate_flat), bool),
             "outside_old_patches": data["new"].ravel()[candidate_flat]}
    if geo_mask is not None:
        masks["geographic_holdout"] = geo_mask
        masks["geographic_source"] = ~geo_mask
    hit = {scope: {"history": [], **{f"{kind}_{alpha:g}": []
                 for kind in KINDS for alpha in alphas[kind]}}
           for scope in masks}
    daily_positive = {scope: [] for scope in masks}
    day_scores = {kind: [] for kind in ("history", *KINDS)}
    truths = []
    for day in days:
        truth = data["reports"][day].ravel()[candidate_flat]
        base = base_score(data, day)
        scores = {"history": base}
        for kind in KINDS:
            feature = response_features(data, inputs, kind, day)
            scores[kind] = feature @ np.asarray(models[kind]["coefficient"], np.float32)
        for scope, mask in masks.items():
            daily_positive[scope].append(int(truth[mask].sum()))
            hit[scope]["history"].append(float(np.dot(selected(base[mask], .1), truth[mask])))
            for kind in KINDS:
                for alpha in alphas[kind]:
                    key = f"{kind}_{alpha:g}"
                    hit[scope][key].append(float(np.dot(
                        selected((base + alpha * scores[kind])[mask], .1), truth[mask])))
        for kind in scores:
            day_scores[kind].append(scores[kind].astype(np.float32))
        truths.append(truth)
        if day % 90 == 0:
            print("scored", day, flush=True)
    result = {}
    for scope, rows in hit.items():
        total = sum(daily_positive[scope])
        result[scope] = {name: {"candidate_cells": int(masks[scope].sum()),
                                "positive_cell_days": total,
                                "top_10pct_capture": round(float(sum(hits) / total), 5),
                                "daily_positive": daily_positive[scope],
                                "daily_captured": np.round(hits, 5).tolist()}
                         for name, hits in rows.items()}
    return result, np.stack(truths), {key: np.stack(value)
                                    for key, value in day_scores.items()}


def main(geographic_holdout: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_grid()
    inputs, graph_audit = build_inputs(data)
    print("graph", graph_audit, flush=True)
    district_ids = data["district_id"].ravel()[data["candidate_flat"]]
    holdout_ids = {i for i, item in enumerate(data["districts"])
                   if item["code"] in HOLDOUT}
    geo_mask = np.isin(district_ids, list(holdout_ids)) if geographic_holdout else None
    pairs, pair_audit = training_pairs(data, ~geo_mask if geographic_holdout else None)
    print("pairs", pair_audit, flush=True)
    models = {kind: fit_response(data, inputs, pairs, kind) for kind in KINDS}
    print("fit", {k: v["coefficient"] for k, v in models.items()}, flush=True)
    validation, _, _ = score_period(data, inputs, models,
                                    range(VALID_START, TEST_START), geo_mask=geo_mask)
    selection_scope = "geographic_source" if geographic_holdout else "citywide"
    chosen = {kind: max(ALPHAS, key=lambda a: validation[selection_scope][
        f"{kind}_{a:g}"]["top_10pct_capture"]) for kind in KINDS}
    selection = {kind: [chosen[kind]] for kind in KINDS}
    q4, truth, scores = score_period(data, inputs, models,
                                    range(TEST_START, N_DAYS), selection,
                                    geo_mask=geo_mask)
    new = data["new"].ravel()[data["candidate_flat"]]
    base = scores["history"]
    for scope in q4:
        if scope == "citywide" and q4[scope]["history"]["positive_cell_days"] != 7209:
            raise ValueError("Q4 citywide positive count changed")
        for kind in KINDS:
            key = f"{kind}_{chosen[kind]:g}"
            q4[scope][key + "_minus_history_interval"] = paired_interval(
                q4[scope][key], q4[scope]["history"])
        for kind in ("distance", "reverse"):
            key = f"uphill_{chosen['uphill']:g}"
            control = f"{kind}_{chosen[kind]:g}"
            q4[scope]["uphill_minus_" + kind + "_interval"] = paired_interval(
                q4[scope][key], q4[scope][control])
    routed_mask = inputs["uphill"].sum(axis=1) > 0
    routed_only = {
        "note": "Diagnostic top-10% queue restricted to cells receiving at least one 500–3000 m terrain path; different candidate budget from citywide",
        "candidate_cells": int(routed_mask.sum()),
        "positive_cell_days": int(truth[:, routed_mask].sum()),
        "capture": {"history": capture(base, truth, routed_mask)["top_10pct_capture"],
                    **{kind: capture(base + chosen[kind] * scores[kind],
                                     truth, routed_mask)["top_10pct_capture"]
                       for kind in KINDS}},
    }
    district = {}
    for i, item in enumerate(data["districts"]):
        mask = district_ids == i
        district[item["code"]] = {"name": item["name_en"],
            "positive_cell_days": int(truth[:, mask].sum()),
            "capture": {"history": capture(base, truth, mask)["top_10pct_capture"],
                        **{kind: capture(base + chosen[kind] * scores[kind],
                                         truth, mask)["top_10pct_capture"]
                           for kind in KINDS}}}
    payload = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "target": "100 m Traffy report buffer touching a 250 m BMA cell-day; no report is unknown, not verified dry",
               "graph_audit": graph_audit, "training_pairs": pair_audit,
               "train_period": ["2023-01-01", "2024-12-31"],
               "selection_period": ["2025-01-01", "2025-09-30"],
               "selection_scope": selection_scope,
               "geographic_holdout_codes": list(HOLDOUT) if geographic_holdout else [],
               "q4_period": ["2025-10-01", "2025-12-31"],
               "baseline": "log1p(prior365 reports)+0.4log1p(prior90)+0.4log1p(250m Gaussian prior365)",
               "learning": "ridge-regularized pairwise logistic residual with fixed history-score offset; nonnegative rain-response coefficients; positives versus sampled unreported comparison cells, not verified negatives",
               "models": models, "selected_multiplier": chosen,
               "validation_capture": {kind: {f"{a:g}": validation[selection_scope][f"{kind}_{a:g}"]["top_10pct_capture"]
                                            for a in ALPHAS} for kind in KINDS},
               "q4": q4, "q4_routed_cells_only": routed_only, "districts": district,
               "limitations": ["Cached BMA DTM exports have variable 512x512 pixel sizes, not uniform 10 m LiDAR",
                               "D8 terrain paths omit drains, walls, culverts, gates and pumps; catchments truncate at BMA edge",
                               "WorldCover built fraction is only a paved/impervious proxy",
                               "Full local-day rain and peak 3 h can occur after a complaint; results are retrospective",
                               "Open-Meteo rain is sampled at one district proxy per 20 km tile",
                               "Q4 2025 labels have been inspected in earlier work; this is exploratory",
                               "Report-positive capture does not validate flood extent, calibrated probabilities or doubled-rain scenarios"]}
    output = (OUT / "q4_2025_uphill_built_rain_ranker_geographic_holdout.json"
              if geographic_holdout else OUT_FILE)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print("selected", chosen, flush=True)
    print("Q4", {scope: {name: row["top_10pct_capture"] for name, row in records.items()
                         if isinstance(row, dict) and "top_10pct_capture" in row}
                 for scope, records in q4.items()}, flush=True)
    print("saved", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--geographic-holdout", action="store_true")
    main(parser.parse_args().geographic_holdout)
