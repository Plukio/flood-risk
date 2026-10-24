"""Evaluate unfamiliar-site ranking without an imposed discovery-slot quota.

The candidate set is zero-own-history cells in each held-out district/day.
The outcome is a first-in-year 250 m raw-report site medoid. Scores and model
arms were selected on source districts in the existing physical-site study.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, load_grid
from flood_citywide_physical_site_data import build
from flood_citywide_physical_site_density_matrix import GROUPS, density_groups
from flood_citywide_physical_site_replay import BLENDS, group_events, pct
from flood_citywide_physical_site_residual import ARMS, MODEL, score_days
from flood_citywide_history_ranker import VALID_START
from flood_citywide_spatial_novel_queue import SpatialNovelCNN, static_image, top_weights
from flood_multi_target_citywide_data import OUT


RESULT = OUT / "q4_2025_site_novel_rank_curve.json"
EVENTS = OUT / "q4_2025_site_novel_rank_curve_events.csv"
FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def source_mean_percentiles(arrays, score, grouped, district, members):
    """Source-only selection criterion with no operating cutoff or slot reserve."""
    ids_by_district = {int(d): np.flatnonzero(district == d) for d in members}
    local_index = {d: {int(cell): j for j, cell in enumerate(ids)}
                   for d, ids in ids_by_district.items()}
    total = np.zeros(len(BLENDS), np.float64)
    count = 0
    for (day, d), events in grouped.items():
        valid = [candidate for event, candidate in events
                 if event["new_site"] and arrays[3][day, candidate]]
        if not valid:
            continue
        ids = ids_by_district[d]
        eligible = arrays[3][day, ids]
        s = pct(arrays[1][day, ids][eligible])
        c = pct(score[day, ids][eligible])
        eligible_lookup = {int(cell): j for j, cell in enumerate(ids[eligible])}
        target = [eligible_lookup[candidate] for candidate in valid]
        for i, alpha in enumerate(BLENDS):
            total[i] += float(pct((1-alpha)*s+alpha*c)[target].sum())
        count += len(valid)
    if count == 0:
        raise ValueError("No source validation events")
    return {str(alpha): round(float(total[i]/count), 6)
            for i, alpha in enumerate(BLENDS)}, count


def district_interval(rows, metric, seed):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["district_index"]].append(r)
    districts = list(grouped)
    n = np.asarray([len(grouped[d]) for d in districts], np.float64)
    total = np.asarray([sum(metric(r) for r in grouped[d]) for d in districts],
                       np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(districts), size=(5000, len(districts)))
    value = total[draws].sum(axis=1)/n[draws].sum(axis=1)
    return [round(float(x), 5) for x in np.quantile(value, [.025, .975])]


def main():
    torch.set_num_threads(4)
    data = load_grid()
    flat = data["candidate_flat"]
    lookup = np.full(np.prod(SHAPE), -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    district = data["district_id"].ravel()[flat]
    density_group, _, density_detail = density_groups(data["districts"])
    sites = build(data["candidate"], data["district_id"])[250]
    states = torch.load(MODEL, map_location="cpu", weights_only=True)["states"]
    folds = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    rows = []
    selections = []
    for fold, members in enumerate(folds, 1):
        source_members = np.setdiff1d(np.arange(len(data["districts"])), members)
        static, _ = static_image(data, ~np.isin(district, members))
        source_allowed = np.zeros(len(data["districts"]), bool)
        source_allowed[source_members] = True
        val_group = group_events(sites, source_allowed, VALID_START, TEST_START, lookup)
        validation = {}
        for candidate_arm in ARMS:
            candidate_model = SpatialNovelCNN().eval()
            candidate_model.load_state_dict(states[fold][candidate_arm])
            val_arrays, val_score = score_days(
                data, candidate_model, static, range(VALID_START, TEST_START))
            ranks, n_val = source_mean_percentiles(
                val_arrays, val_score, val_group, district, source_members)
            validation[candidate_arm] = {"mean_event_percentile": ranks,
                                         "events": n_val}
            del val_arrays, val_score
            print("fold", fold, "validation arm", candidate_arm,
                  "events", n_val, "best", max(ranks.values()), flush=True)
        arm, alpha = max(((a, x) for a in ARMS for x in BLENDS),
                         key=lambda z: (validation[z[0]]["mean_event_percentile"][str(z[1])],
                                        -ARMS.index(z[0]), -z[1]))
        selections.append({"fold": fold, "arm": arm, "alpha": alpha,
                           "source_validation": validation})
        model = SpatialNovelCNN().eval()
        model.load_state_dict(states[fold][arm])
        arrays, score = score_days(data, model, static, range(TEST_START, N_DAYS))
        allowed = np.zeros(len(data["districts"]), bool)
        allowed[members] = True
        grouped = group_events(sites, allowed, TEST_START, N_DAYS, lookup)
        ids_by_district = {int(d): np.flatnonzero(district == d) for d in members}
        local_index = {d: {int(cell): j for j, cell in enumerate(ids)}
                       for d, ids in ids_by_district.items()}
        for (day, d), events in grouped.items():
            new = [(event, candidate) for event, candidate in events
                   if event["new_site"] and arrays[3][day, candidate]]
            if not new:
                continue
            ids = ids_by_district[d]
            eligible = arrays[3][day, ids]
            n = int(eligible.sum())
            s = arrays[1][day, ids][eligible]
            c = score[day, ids][eligible]
            mixed = (1-alpha)*pct(s)+alpha*pct(c)
            eligible_ids = ids[eligible]
            lookup_eligible = {int(cell): j for j, cell in enumerate(eligible_ids)}
            weights = {f"history_{f:g}": top_weights(s, f*n) for f in FRACTIONS}
            weights.update({f"cnn_{f:g}": top_weights(mixed, f*n) for f in FRACTIONS})
            for event, candidate in new:
                j = lookup_eligible[candidate]
                row = {"fold": fold, "day": event["day"], "district_index": d,
                       "row": event["row"], "col": event["col"],
                       "raw_reports": event["raw_reports"],
                       "nearest_prior_report_m": event["nearest_prior_report_m"],
                       "eligible_cells": n, "history_percentile": float(pct(s)[j]),
                       "cnn_percentile": float(pct(mixed)[j]),
                       "density_group": density_group[d]}
                row.update({name: float(w[j]) for name, w in weights.items()})
                rows.append(row)
        print("fold", fold, "arm", arm, "alpha", alpha,
              "eligible new events", sum(r["fold"] == fold for r in rows), flush=True)
        del arrays, score
    if len(rows) != 393:
        raise ValueError(f"Expected 393 first-in-year, zero-own-history sites; got {len(rows)}")
    with EVENTS.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n = len(rows)
    curve = []
    for f in FRACTIONS:
        a = sum(r[f"history_{f:g}"] for r in rows)
        b = sum(r[f"cnn_{f:g}"] for r in rows)
        curve.append({"fraction_of_zero_history_cells": f,
                      "history_hits": round(a, 4), "cnn_hits": round(b, 4),
                      "history_capture_pct": round(100*a/n, 3),
                      "cnn_capture_pct": round(100*b/n, 3),
                      "delta_points": round(100*(b-a)/n, 3)})
    bands = []
    for group in GROUPS:
        subset = [r for r in rows if r["density_group"] == group]
        bands.append({"density_group": group, "events": len(subset),
                      "history_mean_percentile": round(float(np.mean(
                          [r["history_percentile"] for r in subset])), 5),
                      "cnn_mean_percentile": round(float(np.mean(
                          [r["cnn_percentile"] for r in subset])), 5),
                      "delta_percentile_points": round(100*float(np.mean(
                          [r["cnn_percentile"]-r["history_percentile"] for r in subset])), 3),
                      "prior_density_range": density_detail[group]})
    distance = []
    for label, low, high in (("250-500m", 250, 500),
                             ("500-750m", 500, 750),
                             (">750m", 750, float("inf"))):
        subset = [r for r in rows if low < r["nearest_prior_report_m"] <= high]
        distance.append({"distance_band": label, "events": len(subset),
                         "history_mean_percentile": round(float(np.mean(
                             [r["history_percentile"] for r in subset])), 5) if subset else None,
                         "cnn_mean_percentile": round(float(np.mean(
                             [r["cnn_percentile"] for r in subset])), 5) if subset else None})
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "task": "Rank first-in-year reported sites among zero-own-history candidate cells, without a fixed discovery-slot reserve",
              "selection": "CNN arm and nearby-history blend maximize mean new-site rank percentile on Jan-Sep 2025 source districts, without a shortlist cutoff",
              "fold_selections": selections,
              "candidate_site_events": n, "candidate_cells": "Zero own-cell reports in preceding 365 days, separately for each district/day",
              "mean_history_percentile": round(float(np.mean([r["history_percentile"] for r in rows])), 5),
              "mean_cnn_percentile": round(float(np.mean([r["cnn_percentile"] for r in rows])), 5),
              "mean_percentile_difference_district_interval": district_interval(
                  rows, lambda r: r["cnn_percentile"]-r["history_percentile"], 804),
              "curve": curve,
              "density_bands": bands,
              "distance_bands": distance,
              "limits": ["Novel-candidate ranking does not specify a field allocation",
                         "The 10% whole-district operational queue may favor familiar sites",
                         "Report medoids are not verified flood extent",
                         "Q4 2025 was repeatedly inspected during model development"]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
