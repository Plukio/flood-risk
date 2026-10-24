"""Create matched spatial-context pairs from BMA pipe connectivity.

This is a synthetic relation target: same snapped pipe component versus a
different component, at the same map distance from a target cell. It is not an
observed water-flow or flood label.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from flood_citywide_drainage_network import SNAPSHOT, endpoint_graph, geometry, sampled_pipe_points
from flood_citywide_history_cnn_transfer import RADIUS
from flood_citywide_history_ranker import SHAPE, load_grid
from flood_citywide_route_pretrain import terrain
from flood_citywide_surface_osm import build as buildings
from flood_multi_target_citywide_data import OUT, X0, Y0

PAIRS = OUT / "bma_drainage_spatial_matched_pairs_250m.npz"
AUDIT = OUT / "bma_drainage_spatial_matched_pairs_250m_audit.json"
MAX_PAIRS_PER_TARGET = 4
MAX_ELEVATION_GAP_M = .75
MAX_BUILT_GAP = .35


def offsets():
    by_distance = defaultdict(list)
    for dr in range(-RADIUS, RADIUS+1):
        for dc in range(-RADIUS, RADIUS+1):
            squared = dr*dr+dc*dc
            if squared >= 4:
                by_distance[squared].append((dr, dc))
    return by_distance


def build(data=None):
    data = load_grid() if data is None else data
    flat = data["candidate_flat"]
    if PAIRS.exists() and AUDIT.exists():
        with np.load(PAIRS) as saved:
            if not np.array_equal(saved["candidate_flat"], flat):
                raise ValueError("Cached spatial pairs use a different grid")
            pairs = saved["pairs"]
        return pairs, json.loads(AUDIT.read_text())
    with gzip.open(SNAPSHOT, "rt") as handle:
        raw = json.load(handle)
    pipes = geometry(raw["layers"]["5"])
    _, nodes, lengths, graph, _ = endpoint_graph(pipes)
    n_components, component = connected_components(graph, directed=False)
    h, w = SHAPE
    rr, cc = np.indices(SHAPE)
    centers = np.column_stack((X0+(cc.ravel()+.5)*250.,
                               Y0+(h-rr.ravel()-.5)*250.))
    points, owner = sampled_pipe_points(pipes, lengths, np.arange(len(pipes)))
    near, index = cKDTree(points).query(centers[flat], workers=-1)
    attached = near <= 250.
    comp = component[nodes[2*owner[index]]]
    lookup = np.full(h*w, -1, np.int32)
    lookup[flat] = np.arange(len(flat), dtype=np.int32)
    z, _, _, _ = terrain(data)
    elevation = z.ravel()[flat]
    footprint = buildings().ravel()[flat]
    cover = data["static"][:, 7]
    impervious = np.clip(.6*cover+.4*footprint, 0, 1)
    options = offsets()
    rng = np.random.default_rng(861)
    pairs = []
    elev_gaps, built_gaps = [], []
    positive_targets = 0
    for target in np.flatnonzero(attached):
        tr, tc = divmod(int(flat[target]), w)
        positives = []
        for squared, ring in options.items():
            for dr, dc in ring:
                sr, sc = tr+dr, tc+dc
                if not (0 <= sr < h and 0 <= sc < w):
                    continue
                source = lookup[sr*w+sc]
                if source >= 0 and attached[source] and comp[source] == comp[target]:
                    positives.append((int(source), dr, dc, squared))
        if not positives:
            continue
        positive_targets += 1
        for source, pdr, pdc, squared in rng.permutation(positives)[:MAX_PAIRS_PER_TARGET]:
            eligible = []
            for ndr, ndc in options[squared]:
                nr, nc = tr+ndr, tc+ndc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                negative = lookup[nr*w+nc]
                if negative < 0 or not attached[negative] or comp[negative] == comp[target]:
                    continue
                dz = abs(float(elevation[negative]-elevation[source]))
                db = abs(float(impervious[negative]-impervious[source]))
                if dz <= MAX_ELEVATION_GAP_M and db <= MAX_BUILT_GAP:
                    eligible.append((dz/MAX_ELEVATION_GAP_M+
                                     db/MAX_BUILT_GAP, int(negative), ndr, ndc, dz, db))
            if not eligible:
                continue
            _, negative, ndr, ndc, dz, db = min(eligible)
            pairs.append((target, source, negative, pdr, pdc, ndr, ndc))
            elev_gaps.append(dz)
            built_gaps.append(db)
    pairs = np.asarray(pairs, np.int32).reshape(-1, 7)
    if len(pairs) < 2000:
        raise ValueError(f"Too few network spatial pairs: {len(pairs)}")
    AUDIT.write_text(json.dumps({
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "pipe_components": n_components, "attached_candidate_cells": int(attached.sum()),
        "targets_with_connected_source_2_to_7_cells_away": positive_targets,
        "matched_pairs": len(pairs),
        "positive_rule": "source and target nearest pipes share a 5m-snapped component",
        "negative_rule": "source nearest pipe is a different component",
        "source_distance_match": "exact squared 250m-grid distance from same target",
        "maximum_source_elevation_gap_m": MAX_ELEVATION_GAP_M,
        "median_source_elevation_gap_m": float(np.median(elev_gaps)),
        "maximum_source_built_gap": MAX_BUILT_GAP,
        "median_source_built_gap": float(np.median(built_gaps)),
        "built_match": "0.6 WorldCover built fraction + 0.4 OSM building footprint fraction",
        "maximum_pairs_per_target": MAX_PAIRS_PER_TARGET,
        "limitations": ["Nearest pipe is not a verified street inlet",
                        "Shared snapped component is not confirmed hydraulic connectivity",
                        "Path may leave the 15x15 model patch",
                        "No pump/gate state or observed flow direction"]}, indent=2)+"\n")
    np.savez_compressed(PAIRS, pairs=pairs, candidate_flat=flat)
    return pairs, json.loads(AUDIT.read_text())


if __name__ == "__main__":
    print(json.dumps(build()[1], indent=2), flush=True)
