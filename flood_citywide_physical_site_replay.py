"""Replay frozen district queues against raw-point report-site event proxies.

The original models and physical-site radius are not selected on Q4 outcomes.
Only the blend of nearby history and the saved CNN is chosen on Jan-Sep 2025
source districts, then tested in excluded districts in Q4. No scored-day
report enters a model input. This evaluates reported sites, not wet ground.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.sparse import diags
from scipy.stats import rankdata

from flood_citywide_connected_score_smoother import messages
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, VALID_START, load_grid
from flood_citywide_physical_site_data import RADII, build
from flood_citywide_spatial_mask_augmentation import OUT_MODEL
from flood_citywide_spatial_novel_queue import (SpatialNovelCNN, collect,
                                                 queue_weights, static_image)
from flood_citywide_terrain_continuity import links
from flood_multi_target_citywide_data import OUT


MODEL_RESULT = OUT / "q4_2025_mask_cnn_connected_score_smoother_fivefold.json"
RESULT = OUT / "q4_2025_physical_site_replay.json"
SITES = OUT / "q4_2025_physical_site_replay_events_250m.csv"
BLENDS = (0., .25, .5, .75, 1.)
QUOTA = .20


def group_events(events: list[dict], allowed: np.ndarray,
                 begin: int, end: int, flat_to_candidate: np.ndarray):
    grouped = defaultdict(list)
    for event in events:
        day = event["day_index"]
        if begin <= day < end and allowed[event["district_index"]]:
            candidate = int(flat_to_candidate[event["flat"]])
            if candidate >= 0:
                grouped[(day-begin, event["district_index"])].append(
                    (event, candidate))
    return grouped


def pct(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average").astype(np.float32)/max(1, len(values))


def event_capture(arrays, score, grouped, district_id, members, radius,
                  alpha: float, event_rows: bool = False):
    base, simple, _, novel, truth = arrays
    totals = {"all": [0, 0.], "new_site": [0, 0.],
              "under_100m": [0, 0.], "100_to_250m": [0, 0.],
              "250_to_750m": [0, 0.], "over_750m": [0, 0.],
              "raster_positive": [0, 0.]}
    rows = []
    ids_by_district = {int(i): np.flatnonzero(district_id == i) for i in members}
    local_index = {int(i): {int(cell): j for j, cell in enumerate(ids)}
                   for i, ids in ids_by_district.items()}
    for (local_day, district), events in grouped.items():
        ids = ids_by_district[district]
        b, s, c, n = (base[local_day, ids], simple[local_day, ids],
                      score[local_day, ids], novel[local_day, ids])
        rank_s, rank_c = np.zeros(len(ids), np.float32), np.zeros(len(ids), np.float32)
        rank_s[n], rank_c[n] = pct(s[n]), pct(c[n])
        mixed = (1-alpha)*rank_s + alpha*rank_c
        w0 = queue_weights(b, s, n, QUOTA)
        w1 = queue_weights(b, mixed, n, QUOTA)
        if abs(float(w0.sum())-.1*len(ids)) > .02 or abs(float(w1.sum())-.1*len(ids)) > .02:
            raise ValueError("Daily district inspection budget changed")
        for event, candidate in events:
            j = local_index[district][candidate]
            d = event["nearest_prior_report_m"]
            band = ("under_100m" if d <= 100 else
                    "100_to_250m" if d <= 250 else
                    "250_to_750m" if d <= 750 else "over_750m")
            for key in ("all", band, "new_site" if event["new_site"] else None,
                        "raster_positive" if truth[local_day, candidate] else None):
                if key:
                    totals[key][0] += 1
                    totals[key][1] += float(w1[j])
            if event_rows:
                rows.append({"day": event["day"], "district_index": district,
                             "row": event["row"], "col": event["col"],
                             "raw_reports": event["raw_reports"],
                             "nearest_prior_report_m": round(d, 2),
                             "new_site_at_radius": int(event["new_site"]),
                             "own_cell_history_zero": int(n[j]),
                             "raster_positive": int(truth[local_day, candidate]),
                             "nearby_history_selection_weight": round(float(w0[j]), 6),
                             "model_selection_weight": round(float(w1[j]), 6)})
    return {key: {"events": value[0], "model_hits": round(value[1], 5),
                  "model_capture_pct": round(100*value[1]/value[0], 3) if value[0] else None}
            for key, value in totals.items()}, rows


def evaluate_blends(arrays, score, grouped, district_id, members, radius):
    # Preserve exact nearby-history score at alpha=0; percentile blending is
    # used only when alpha>0, and the baseline is reported separately below.
    output = {}
    for alpha in BLENDS:
        result, _ = event_capture(arrays, score, grouped, district_id, members,
                                  radius, alpha)
        output[str(alpha)] = result
    return output


def baseline_capture(arrays, grouped, district_id, members):
    base, simple, _, novel, truth = arrays
    ids_by_district = {int(i): np.flatnonzero(district_id == i) for i in members}
    local_index = {int(i): {int(cell): j for j, cell in enumerate(ids)}
                   for i, ids in ids_by_district.items()}
    out = defaultdict(lambda: [0, 0.])
    for (local_day, district), events in grouped.items():
        ids = ids_by_district[district]
        weight = queue_weights(base[local_day, ids], simple[local_day, ids],
                               novel[local_day, ids], QUOTA)
        for event, candidate in events:
            w = float(weight[local_index[district][candidate]])
            d = event["nearest_prior_report_m"]
            band = ("under_100m" if d <= 100 else
                    "100_to_250m" if d <= 250 else
                    "250_to_750m" if d <= 750 else "over_750m")
            for key in ("all", band, "new_site" if event["new_site"] else None):
                if key:
                    out[key][0] += 1
                    out[key][1] += w
    return {k: {"events": v[0], "hits": round(v[1], 5),
                "capture_pct": round(100*v[1]/v[0], 3) if v[0] else None}
            for k, v in out.items()}


def main():
    torch.set_num_threads(4)
    data = load_grid()
    event_sets = build(data["candidate"], data["district_id"])
    flat = data["candidate_flat"]
    lookup = np.full(np.prod(SHAPE), -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    district = data["district_id"].ravel()[flat]
    ordinary, _, _, _, _, _ = links(data)
    degree = np.asarray(ordinary.sum(axis=1)).ravel()
    graph = diags(np.divide(1, degree, out=np.zeros_like(degree), where=degree > 0)) @ ordinary
    saved = torch.load(OUT_MODEL, map_location="cpu", weights_only=True)["states"]
    prior = json.loads(MODEL_RESULT.read_text())
    folds = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    all_rows, summaries = [], []
    for number, members in enumerate(folds, 1):
        source = np.setdiff1d(np.arange(len(data["districts"])), members)
        target_mask = np.isin(district, members)
        static, _ = static_image(data, ~target_mask)
        model = SpatialNovelCNN().eval()
        model.load_state_dict(saved[number]["synthetic_center_block"])
        choice = prior["folds"][number-1]["selected_on_source"]["ordinary"]

        def predict(days):
            arrays = collect(data, model, static, days)
            score = arrays[2]
            if choice["hops"]:
                for hops, message in messages(arrays[2], arrays[3], graph):
                    if hops == choice["hops"]:
                        score = (1-choice["alpha"])*score+choice["alpha"]*message
                        break
            return arrays, score

        val_arrays, val_score = predict(range(VALID_START, TEST_START))
        source_allowed = np.zeros(len(data["districts"]), bool)
        source_allowed[source] = True
        val250 = group_events(event_sets[250], source_allowed, VALID_START,
                              TEST_START, lookup)
        val = evaluate_blends(val_arrays, val_score, val250, district, source, 250)
        alpha = max(BLENDS, key=lambda a: (val[str(a)]["new_site"]["model_capture_pct"], -a))
        del val_arrays, val_score
        q4_arrays, q4_score = predict(range(TEST_START, N_DAYS))
        target_allowed = np.zeros(len(data["districts"]), bool)
        target_allowed[members] = True
        by_radius = {}
        for radius in RADII:
            grouped = group_events(event_sets[radius], target_allowed,
                                   TEST_START, N_DAYS, lookup)
            baseline = baseline_capture(q4_arrays, grouped, district, members)
            model_result, rows = event_capture(q4_arrays, q4_score, grouped,
                                                district, members, radius, alpha,
                                                event_rows=radius == 250)
            if radius == 250:
                for row in rows:
                    row["fold"] = number
                all_rows += rows
            by_radius[str(radius)] = {"baseline": baseline,
                                      "source_selected_blend": model_result}
        summaries.append({"fold": number,
                          "district_codes": [data["districts"][i]["code"] for i in members],
                          "source_validation_250m": val,
                          "source_selected_alpha": alpha,
                          "q4": by_radius})
        print("fold", number, "alpha", alpha,
              "250m new", by_radius["250"]["baseline"].get("new_site"),
              by_radius["250"]["source_selected_blend"]["new_site"], flush=True)
    with SITES.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    pooled = {}
    for radius in RADII:
        pooled[str(radius)] = {}
        for arm in ("baseline", "source_selected_blend"):
            pooled[str(radius)][arm] = {}
            for key in ("all", "new_site", "under_100m", "100_to_250m",
                        "250_to_750m", "over_750m"):
                rows = [f["q4"][str(radius)][arm].get(key, {}) for f in summaries]
                count = sum(r.get("events", 0) for r in rows)
                hits = sum(r.get("hits", r.get("model_hits", 0)) for r in rows)
                pooled[str(radius)][arm][key] = {
                    "events": count, "hits": round(hits, 4),
                    "capture_pct": round(100*hits/count, 3) if count else None}
    output = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "task": "Site-event medoid cell capture at the same daily top-10% per-district queue, 20% novel-slot reserve",
              "site_label": "Same-day complete-linkage point clusters; no report within radius on earlier day in prior 365 days",
              "blend": "Percentile ranks of nearby-history and frozen CNN among zero-own-history cells only; alpha chosen on 250m source-district Jan-Sep new sites",
              "folds": summaries, "pooled_q4": pooled,
              "limits": ["Q4 2025 was repeatedly inspected; exploratory",
                         "No report is not observed dry ground",
                         "A site event is a coordinate proxy, not surveyed flood extent",
                         "The medoid cell may not equal every cell in the old 100m buffer target"]}
    RESULT.write_text(json.dumps(output, indent=2)+"\n")
    print("pooled", json.dumps(pooled["250"], indent=2), flush=True)


if __name__ == "__main__":
    main()
