"""Route masked-CNN report scores through plausible 250 m surface connections.

This is a report-ranking experiment, not a flood-extent or depth model. A
building/cover fraction attenuates a possible route; it does not prove a wall
or an open passage. Path settings are chosen on source-district validation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags

from flood_citywide_history_ranker import N_DAYS, TEST_START, VALID_START, load_grid
from flood_citywide_spatial_mask_augmentation import OUT_MODEL
from flood_citywide_spatial_novel_queue import SpatialNovelCNN, collect, static_image, top_weights
from flood_citywide_terrain_continuity import links
from flood_citywide_route_pretrain import terrain
from flood_citywide_surface_osm import build as building_fraction
from flood_citywide_target_site_mask_test import q20_rows
from flood_multi_target_citywide_data import OUT
from flood_open_weak_data import FEATURE_NAMES

OUTPUT = OUT / "q4_2025_mask_cnn_connected_score_smoother_fivefold.json"
HOPS = (1, 2, 3)
ALPHAS = (0., .25, .5, 1.)
KINDS = ("ordinary", "terrain", "terrain_reverse", "terrain_build", "terrain_build_cover",
         "ordinary_score_gate", "terrain_reverse_score_gate", "terrain_build_cover_score_gate")


def graph_family(data):
    _, _, a, b, _, audit = links(data)
    z, _, _, _ = terrain(data)
    flat = data["candidate_flat"]
    z = z.ravel()[flat]
    building = building_fraction().ravel()[flat]
    static = data["static"]
    cover = (static[:, FEATURE_NAMES.index("worldcover_crop_fraction_2021")] +
             static[:, FEATURE_NAMES.index("worldcover_tree_grass_fraction_2021")])
    # Each pair contributes both possible directions. Row is the cell
    # receiving a message; column is the predicted source cell.
    target = np.r_[a, b]
    source = np.r_[b, a]
    rise = z[target] - z[source]
    gap = np.abs(rise)
    ordinary = np.ones(len(target), np.float32)
    # An uphill move needs an unknown water head. Near-flat and downhill moves
    # remain possible; large mean-height changes receive less weight.
    downhill = np.exp(-np.maximum(rise, 0)/.15 - gap/.75).astype(np.float32)
    reverse = np.exp(-np.maximum(-rise, 0)/.15 - gap/.75).astype(np.float32)
    build = np.exp(-2*np.maximum(building[target], building[source])).astype(np.float32)
    cover_attenuation = np.exp(-.8*(cover[target]+cover[source])*.5).astype(np.float32)
    weights = {"ordinary": ordinary, "terrain": downhill,
               "terrain_reverse": reverse,
               "terrain_build": downhill*build,
               "terrain_build_cover": downhill*build*cover_attenuation}
    graphs = {}
    for kind, weight in weights.items():
        adjacency = csr_matrix((weight, (target, source)), shape=(len(flat), len(flat)))
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        graphs[kind] = diags(np.divide(1, degree, out=np.zeros_like(degree),
                                       where=degree > 0)) @ adjacency
    for kind in ("ordinary", "terrain_reverse", "terrain_build_cover"):
        graphs[f"{kind}_score_gate"] = graphs[kind]
    audit.update({"direction": "source-to-target, uphill attenuated by 0.15 m scale, absolute mean-height gap by 0.75 m scale",
                  "building": "OSM 2025 footprint area fraction in each 250 m cell; max endpoint fraction attenuates by exp(-2*fraction)",
                  "cover": "WorldCover 2021 crop+tree/grass area fraction; mean endpoint fraction attenuates by exp(-0.8*fraction); not measured soil infiltration",
                  "low_head_or_barrier_status": "unknown; weights are soft proxies, never certified open or closed edges",
                  "mean_building_fraction": round(float(building.mean()), 5),
                  "mean_vegetated_fraction": round(float(cover.mean()), 5),
                  "candidate_edges": int(len(a))})
    return graphs, audit


def messages(score, novel, graph, gate_scores=False):
    """Average over 1-3 hops; optionally require high-scored path cells."""
    valid = novel.astype(np.float32)
    weighted = score*valid
    numer, denom = weighted.T.copy(), valid.T.copy()
    if gate_scores:
        # Uncalibrated CNN logits can be compared by their daily rank only.
        # The bottom half of cells cannot relay a message. Among the upper
        # half, relay strength increases linearly with the rank percentile.
        order = np.argsort(score, axis=1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(score.shape[1])[None, :], axis=1)
        percentile = rank.astype(np.float32)/max(1, score.shape[1]-1)
        confidence = np.clip(2*percentile-1, 0, 1).T
    for hops in HOPS:
        if gate_scores:
            numer = numer*confidence
            denom = denom*confidence
        numer = graph @ numer
        denom = graph @ denom
        yield hops, np.divide(numer, denom, out=score.T.copy(), where=denom > 1e-8).T


def self_check():
    # A score can cross an intermediate cell that is not a novel report
    # candidate. This is a connectivity check, not a flood-label assumption.
    adjacency = csr_matrix((np.ones(6), ([0, 1, 1, 2, 2, 3],
                                          [1, 0, 2, 1, 3, 2])), shape=(4, 4))
    graph = diags(1/np.asarray(adjacency.sum(axis=1)).ravel()) @ adjacency
    path = list(messages(np.asarray([[1., 0., 0., 0.]], np.float32),
                         np.asarray([[True, False, False, True]]), graph))
    if path[0][1][0, 3] != 0 or path[2][1][0, 3] <= 0:
        raise ValueError("Multi-hop score did not cross the known-history bridge")
    blocked = list(messages(np.asarray([[1., 0., 0., 0.]], np.float32),
                            np.asarray([[True, False, False, True]]), graph,
                            gate_scores=True))
    if blocked[2][1][0, 3] != 0:
        raise ValueError("Low-scored intermediate cell relayed through the gate")


def novel_capture(score, arrays, scope):
    _, _, _, novel, truth = arrays
    indices = np.flatnonzero(scope)
    budget = .1*len(indices)
    hits, positive = 0., 0
    for day in range(len(truth)):
        candidate = indices[novel[day, indices]]
        if not len(candidate):
            continue
        known_count = len(indices)-len(candidate)
        # Match queue_weights: unused repeat-site slots spill into novelty.
        novel_budget = .2*budget + max(0., .8*budget-known_count)
        y = truth[day, candidate]
        positive += int(y.sum())
        hits += float(np.dot(top_weights(score[day, candidate], min(novel_budget, len(candidate))), y))
    return hits/positive if positive else 0.


def evaluate(data, arrays, score, scope):
    base, simple, _, novel, truth = arrays
    row = q20_rows(data, (base, simple, score, novel, truth), scope, False)["model"]
    return {k: row[k] for k in ("positive_cell_days", "top_10pct_capture",
                              "novel_positive_cell_days", "novel_capture",
                              "daily_positive", "daily_captured", "daily_novel_positive",
                              "daily_novel_captured")}


def pool(folds, kind):
    rows = [f["q4"][kind] for f in folds]
    result = {"positive_cell_days": sum(r["positive_cell_days"] for r in rows),
              "novel_positive_cell_days": sum(r["novel_positive_cell_days"] for r in rows)}
    for field in ("daily_positive", "daily_captured", "daily_novel_positive", "daily_novel_captured"):
        result[field] = np.sum([r[field] for r in rows], axis=0).round(5).tolist()
    result["top_10pct_capture"] = round(sum(result["daily_captured"])/result["positive_cell_days"], 5)
    result["novel_capture"] = round(sum(result["daily_novel_captured"])/result["novel_positive_cell_days"], 5)
    return result


def interval(a, b, novel=True, seed=2305):
    positives = np.asarray(a["daily_novel_positive" if novel else "daily_positive"])
    x = np.asarray(a["daily_novel_captured" if novel else "daily_captured"])
    y = np.asarray(b["daily_novel_captured" if novel else "daily_captured"])
    blocks = [np.arange(i, min(i+7, len(positives))) for i in range(0, len(positives), 7)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        ix = np.concatenate([blocks[k] for k in rng.integers(len(blocks), size=len(blocks))])
        draws.append((x[ix].sum()-y[ix].sum())/positives[ix].sum())
    return np.quantile(draws, [.025, .975]).round(5).tolist()


def select(data, arrays, source, graphs):
    score, novel = arrays[2], arrays[3]
    chosen, table = {}, {}
    for kind, graph in graphs.items():
        best = (novel_capture(score, arrays, source), 0, 0.)
        for hops, message in messages(score, novel, graph,
                                      gate_scores=kind.endswith("_score_gate")):
            for alpha in ALPHAS[1:]:
                adjusted = ((1-alpha)*score+alpha*message).astype(np.float32)
                capture = novel_capture(adjusted, arrays, source)
                table[f"{kind}_{hops}_{alpha:g}"] = round(capture, 5)
                # Ties prefer less propagation, then less weight.
                if capture > best[0] + 1e-9:
                    best = (capture, hops, alpha)
        chosen[kind] = {"hops": best[1], "alpha": best[2],
                        "source_novel_capture": round(best[0], 5)}
    return chosen, table


def main():
    torch.set_num_threads(4)
    self_check()
    data = load_grid()
    graphs, graph_audit = graph_family(data)
    saved = torch.load(OUT_MODEL, weights_only=True)["states"]
    district = data["district_id"].ravel()[data["candidate_flat"]]
    splits = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    folds = []
    for number, ids in enumerate(splits, 1):
        source = ~np.isin(district, ids)
        target = ~source
        static, _ = static_image(data, source)
        model = SpatialNovelCNN().eval()
        model.load_state_dict(saved[number]["synthetic_center_block"])
        print("fold", number, "validation", flush=True)
        val = collect(data, model, static, range(VALID_START, TEST_START))
        choices, table = select(data, val, source, graphs)
        del val
        print("fold", number, "selected", choices, "Q4", flush=True)
        q4_arrays = collect(data, model, static, range(TEST_START, N_DAYS))
        original = evaluate(data, q4_arrays, q4_arrays[2], target)
        q4 = {"original": original}
        for kind, graph in graphs.items():
            choice = choices[kind]
            if not choice["hops"]:
                q4[kind] = original
                continue
            for hops, message in messages(q4_arrays[2], q4_arrays[3], graph,
                                          gate_scores=kind.endswith("_score_gate")):
                if hops == choice["hops"]:
                    adjusted = (1-choice["alpha"])*q4_arrays[2]+choice["alpha"]*message
                    q4[kind] = evaluate(data, q4_arrays, adjusted, target)
                    break
        folds.append({"fold": number, "selected_on_source": choices,
                      "source_validation_novel_capture": table, "q4": q4})
        print("fold", number, "Q4 novel", {k:v["novel_capture"] for k,v in q4.items()}, flush=True)
    pooled = {kind: pool(folds, kind) for kind in ("original", *KINDS)}
    if (pooled["original"]["positive_cell_days"] != 7209 or
        pooled["original"]["novel_positive_cell_days"] != 1664 or
        pooled["original"]["novel_capture"] != .13529):
        raise ValueError("Original held-out queue changed")
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "task": "same held-out Q4 2025 first-in-a-year report queue; 10% total area, 20% novel slot reserve",
              "model": "saved masked CNN plus source-selected connected-path score propagation",
              "selection": "Jan-Sep source districts choose up to 3 hops and alpha 0/.25/.5/1; no Q4 tuning",
              "score_gate": "gated variants let only the top daily half of raw CNN logits relay, with linear percentile weight; fixed before Q4, not flood probabilities",
              "graph_audit": graph_audit, "folds": folds, "pooled_q4": pooled,
              "interval_terrain_build_cover_minus_ordinary": interval(pooled["terrain_build_cover"], pooled["ordinary"]),
              "interval_terrain_build_cover_minus_original": interval(pooled["terrain_build_cover"], pooled["original"]),
              "interval_terrain_build_cover_score_gate_minus_ordinary_score_gate": interval(pooled["terrain_build_cover_score_gate"], pooled["ordinary_score_gate"]),
              "limitations": ["Masked CNN logits rank reported sites, not actual wet cells",
                              "250 m mean DTM misses edge saddles and small barriers",
                              "Building and cover fractions are proxies, not verified edge blockages or measured soil infiltration",
                              "Unknown pump, pipe, gate and water-surface state",
                              "Q4 was previously examined"]}
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    print("pooled", {k:(v["novel_capture"], v["top_10pct_capture"]) for k,v in pooled.items()}, flush=True)
    print("saved", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
