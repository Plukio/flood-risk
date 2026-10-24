"""Build cached 2021 BMA drainage-network features on the 250 m city grid.

Endpoint snaps and pipe-to-canal links are geometric hypotheses, not verified
hydraulic connections or operating outfalls. No report labels enter this file.
"""
from __future__ import annotations

import gzip
import json
import math
from datetime import datetime, timezone

import numpy as np
import requests
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from flood_bangkok_drainage_gis_audit import SERVICE, name, number, query
from flood_citywide_history_ranker import SHAPE
from flood_multi_target_citywide_data import OUT, X0, Y0

SNAPSHOT = OUT / "bma_drainage_citywide_2021_raw.json.gz"
FEATURES = OUT / "bma_drainage_citywide_2021_network_250m.npz"
AUDIT = OUT / "bma_drainage_citywide_2021_network_audit.json"
SNAP_M = 5.
OUTLET_M = 20.
MAX_PATH_M = 5000.


def download() -> dict:
    if SNAPSHOT.exists():
        with gzip.open(SNAPSHOT, "rt") as handle:
            return json.load(handle)
    session = requests.Session()
    rows = {}
    for layer in (5, 6, 7, 1):
        # The public service has a 1000-feature transfer cap. The paged query
        # audits errors and sorts on a stable object ID.
        rows[str(layer)] = query(session, layer, "1=1")
        print("downloaded drainage layer", layer, len(rows[str(layer)]), flush=True)
    data = {"service": SERVICE, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "layers": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(SNAPSHOT, "wt", compresslevel=5) as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
    return data


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)

    def root(self, i: int) -> int:
        p = self.parent
        while p[i] != i:
            p[i] = p[p[i]]
            i = int(p[i])
        return i

    def join(self, a: int, b: int) -> None:
        x, y = self.root(a), self.root(b)
        if x != y:
            self.parent[y] = x


def geometry(rows: list[dict]):
    usable = []
    for row in rows:
        paths = row.get("geometry", {}).get("paths", [])
        if len(paths) != 1 or len(paths[0]) < 2:
            continue
        xy = np.asarray(paths[0], dtype=np.float64)
        if not np.isfinite(xy).all():
            continue
        usable.append((xy, row["attributes"]))
    return usable


def outlet_indices(pipes, canals):
    lines, names = [], []
    for xy, attrs in canals:
        lines.append(LineString(xy))
        names.append(name(attrs.get("KHONG_NAME")))
    tree = STRtree(lines)
    outlet = []
    named = near = 0
    for i, (xy, attrs) in enumerate(pipes):
        target = name(attrs.get("TO_ROAD"))
        if "คลอง" not in target and "แม่น้ำ" not in target:
            continue
        named += 1
        point = Point(xy[-1])
        j = int(tree.nearest(point))
        if point.distance(lines[j]) > OUTLET_M:
            continue
        near += 1
        if target == names[j]:
            outlet.append(2*i+1)
    return np.asarray(outlet, np.int32), {"named_pipe_ends": named,
        "near_canals_20m": near, "named_and_near_candidate_ends": len(outlet)}


def endpoint_graph(pipes):
    endpoints = np.asarray([p[0][j] for p in pipes for j in (0, -1)])
    uf = UnionFind(len(endpoints))
    pairs = cKDTree(endpoints).query_pairs(SNAP_M, output_type="ndarray")
    for a, b in pairs:
        uf.join(int(a), int(b))
    roots = np.fromiter((uf.root(i) for i in range(len(endpoints))),
                        dtype=np.int32, count=len(endpoints))
    _, nodes = np.unique(roots, return_inverse=True)
    nodes = nodes.astype(np.int32)
    lengths = np.array([LineString(xy).length for xy, _ in pipes], np.float64)
    start, end = nodes[::2], nodes[1::2]
    valid = start != end
    rr = np.r_[start[valid], end[valid]]
    cc = np.r_[end[valid], start[valid]]
    ww = np.r_[lengths[valid], lengths[valid]]
    graph = coo_matrix((ww, (rr, cc)), shape=(nodes.max()+1,)*2).tocsr()
    return endpoints, nodes, lengths, graph, len(pairs)


def sampled_pipe_points(pipes, lengths, pipe_indices):
    # Along-line samples let each grid center attach to nearby mapped pipe
    # geometry. The 50 m step is far below the model's 250 m cell width.
    chunks, owners = [], []
    for i, (xy, _) in enumerate(pipes):
        segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.r_[0, np.cumsum(segment)]
        if cumulative[-1] == 0:
            continue
        steps = np.linspace(0, cumulative[-1], max(2, math.ceil(lengths[i]/50)+1))
        j = np.minimum(np.searchsorted(cumulative, steps, side="right")-1,
                       len(segment)-1)
        fraction = (steps-cumulative[j])/np.maximum(segment[j], 1e-9)
        chunks.append(xy[j] + fraction[:, None]*(xy[j+1]-xy[j]))
        owners.extend([pipe_indices[i]]*len(steps))
    return np.concatenate(chunks), np.asarray(owners, np.int32)


def distances(nodes, graph, outlet_endpoint):
    source = np.unique(nodes[outlet_endpoint])
    if not len(source):
        raise ValueError("No candidate canal connections")
    # A super-source computes nearest network distance to any candidate.
    n = graph.shape[0]
    rr = np.r_[graph.tocoo().row, np.full(len(source), n)]
    cc = np.r_[graph.tocoo().col, source]
    ww = np.r_[graph.tocoo().data, np.full(len(source), .0001)]
    augmented = coo_matrix((ww, (rr, cc)), shape=(n+1, n+1)).tocsr()
    d = dijkstra(augmented, directed=False, indices=n)[:n]
    pipe_distance = np.minimum(d[nodes[::2]], d[nodes[1::2]])
    return np.minimum(pipe_distance, MAX_PATH_M)


def make() -> dict:
    raw = download()
    pipes = geometry(raw["layers"]["5"])
    canals = geometry(raw["layers"]["6"])
    endpoints, nodes, lengths, graph, snap_pairs = endpoint_graph(pipes)
    outlets, outlet_audit = outlet_indices(pipes, canals)
    pipe_distance = distances(nodes, graph, outlets)

    attrs = [a for _, a in pipes]
    districts = np.asarray([str(a.get("DISTRICT_I") or "") for a in attrs])
    # Place the same number of pseudo-outlets at random pipe ends within each
    # district. This preserves district-level outlet count and all pipe geometry.
    rng = np.random.default_rng(623)
    shuffled = []
    for district in np.unique(districts):
        real = outlets[districts[outlets//2] == district]
        eligible = np.flatnonzero(np.repeat(districts == district, 2))
        if len(real):
            shuffled.extend(rng.choice(eligible, len(real), replace=False))
    shuffled = np.asarray(shuffled, np.int32)
    control_distance = distances(nodes, graph, shuffled)

    height, width = SHAPE
    rr, cc = np.indices(SHAPE)
    centers = np.column_stack((X0+(cc.ravel()+.5)*250.,
                              Y0+(height-rr.ravel()-.5)*250.))
    samples, owner = sampled_pipe_points(pipes, lengths, np.arange(len(pipes)))
    near_m, index = cKDTree(samples).query(centers, workers=-1)
    attached = owner[index]
    # For distant cells no inlet connection is assumed. Pipe distance remains
    # a separate coverage feature; outlet distance is zero when unattached.
    close = near_m <= 250
    true_score = np.zeros(len(centers), np.float32)
    shuffle_score = np.zeros(len(centers), np.float32)
    true_score[close] = np.exp(-pipe_distance[attached[close]]/1000.)
    shuffle_score[close] = np.exp(-control_distance[attached[close]]/1000.)
    pipe_score = np.exp(-near_m/250.).astype(np.float32)
    point_scores = []
    for layer in ("7", "1"):
        points = np.asarray([(row["geometry"]["x"], row["geometry"]["y"])
                             for row in raw["layers"][layer]
                             if "x" in row.get("geometry", {})], np.float64)
        if len(points) == 0:
            raise ValueError(f"No BMA point geometry in layer {layer}")
        distance, _ = cKDTree(points).query(centers, workers=-1)
        point_scores.append(np.exp(-distance/500.).astype(np.float32))
    features = np.stack((pipe_score.reshape(SHAPE), true_score.reshape(SHAPE),
                         shuffle_score.reshape(SHAPE),
                         point_scores[0].reshape(SHAPE),
                         point_scores[1].reshape(SHAPE))).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite network features")
    np.savez_compressed(FEATURES, features=features)
    audit = {"built_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(SNAPSHOT), "service": SERVICE,
        "source_vintage": "BMA pipe layer YEAR=2564 (2021); no event-time state",
        "pipe_records": len(raw["layers"]["5"]), "usable_single_path_pipes": len(pipes),
        "canal_records": len(raw["layers"]["6"]),
        "pump_equipment_records": len(raw["layers"]["7"]),
        "gate_records": len(raw["layers"]["1"]),
        "endpoint_pairs_within_5m": int(snap_pairs),
        "snapped_nodes": int(graph.shape[0]), **outlet_audit,
        "pipe_attached_cells_within_250m": int(close.sum()),
        "cells_reaching_candidate_within_5km": int(((pipe_distance[attached]<MAX_PATH_M)&close).sum()),
        "control_cells_reaching_candidate_within_5km": int(((control_distance[attached]<MAX_PATH_M)&close).sum()),
        "features": ["exp(-nearest pipe sample distance/250m)",
                     "exp(-undirected pipe path to named+near canal endpoint/1000m), zero if >250m from pipe",
                     "same with district-stratified shuffled pseudo-outlets",
                     "exp(-nearest mapped pump-equipment distance/500m)",
                     "exp(-nearest mapped BMA gate distance/500m)"],
        "cautions": ["Endpoint proximity does not prove connectivity",
                     "Candidate canal endpoints are not verified operating outfalls",
                     "Path is undirected; invert fields and pump/gate states are omitted",
                     "Cell-to-pipe attachment is proximity, not a mapped inlet"]}
    AUDIT.write_text(json.dumps(audit, indent=2)+"\n")
    return audit


if __name__ == "__main__":
    print(json.dumps(make(), indent=2), flush=True)
