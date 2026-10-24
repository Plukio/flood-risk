"""Frozen five-fold site-level replay and paired uncertainty summary."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, load_grid
from flood_citywide_physical_site_data import RADII, build
from flood_citywide_physical_site_replay import event_capture, group_events
from flood_citywide_physical_site_residual import (ARMS, MODEL, RESULT as TRAIN_RESULT,
                                                   score_days)
from flood_citywide_spatial_novel_queue import SpatialNovelCNN, static_image
from flood_multi_target_citywide_data import OUT


OLD_SITES = OUT / "q4_2025_physical_site_replay_events_250m.csv"
RESULT = OUT / "q4_2025_physical_site_final.json"
SITES = OUT / "q4_2025_physical_site_final_events_250m.csv"


def event_key(row):
    return (str(row["day"]), int(row["district_index"]), int(row["row"]),
            int(row["col"]), int(row["raw_reports"]),
            round(float(row["nearest_prior_report_m"]), 2))


def summarize(rows, condition):
    subset = [r for r in rows if condition(r)]
    n = len(subset)
    result = {"events": n}
    for key in ("nearby_history", "old_cnn", *ARMS, "source_selected"):
        hits = sum(r[key] for r in subset)
        result[key] = {"hits": round(hits, 4),
                       "capture_pct": round(100*hits/n, 3) if n else None}
    return result


def interval(rows, candidate, reference, unit, seed):
    grouped = defaultdict(list)
    for r in rows:
        grouped[unit(r)].append(r)
    units = list(grouped)
    n_by = np.asarray([len(grouped[u]) for u in units], np.float64)
    delta_by = np.asarray([sum(r[candidate]-r[reference] for r in grouped[u])
                           for u in units], np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(units), size=(5000, len(units)))
    rates = 100*delta_by[sample].sum(axis=1)/n_by[sample].sum(axis=1)
    return [round(float(x), 3) for x in np.quantile(rates, [.025, .975])]


def main():
    torch.set_num_threads(4)
    data = load_grid()
    events = build(data["candidate"], data["district_id"])
    train = json.loads(TRAIN_RESULT.read_text())
    old_rows = list(csv.DictReader(OLD_SITES.open(encoding="utf-8", newline="")))
    if len(old_rows) != 2326:
        raise ValueError("Frozen old-CNN site replay changed")
    flat = data["candidate_flat"]
    lookup = np.full(np.prod(SHAPE), -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    district = data["district_id"].ravel()[flat]
    saved = torch.load(MODEL, map_location="cpu", weights_only=True)["states"]
    folds = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    all_rows, fold_summary = [], []
    for fold, members in enumerate(folds, 1):
        static, _ = static_image(data, ~np.isin(district, members))
        target_allowed = np.zeros(len(data["districts"]), bool)
        target_allowed[members] = True
        source_validation = train["folds"][fold-1]["arms"]
        selected_arm = max(ARMS, key=lambda a: (
            source_validation[a]["source_validation_new_site"][
                str(source_validation[a]["source_selected_alpha"])]["model_capture_pct"],
            -ARMS.index(a)))
        by_arm = {}
        radius_results = {}
        for arm in ARMS:
            model = SpatialNovelCNN().eval()
            model.load_state_dict(saved[fold][arm])
            arrays, score = score_days(data, model, static, range(TEST_START, N_DAYS))
            alpha = source_validation[arm]["source_selected_alpha"]
            by_arm[arm] = {}
            radius_results[arm] = {}
            for radius in RADII:
                grouped = group_events(events[radius], target_allowed,
                                       TEST_START, N_DAYS, lookup)
                result, rows = event_capture(arrays, score, grouped, district,
                                              members, radius, alpha,
                                              event_rows=radius == 250)
                radius_results[arm][str(radius)] = result
                if radius == 250:
                    by_arm[arm] = rows
            del arrays, score
        one = by_arm[ARMS[0]]
        two = by_arm[ARMS[1]]
        if len(one) != len(two):
            raise ValueError("Site event alignment changed between arms")
        frozen = [r for r in old_rows if int(r["fold"]) == fold]
        if len(frozen) != len(one):
            raise ValueError("Site event alignment changed from old replay")
        for a, b, old in zip(one, two, frozen):
            if event_key(a) != event_key(b) or event_key(a) != event_key(old):
                raise ValueError("New model and frozen comparison event keys differ")
            row = {"fold": fold, "day": a["day"],
                   "district_index": int(a["district_index"]),
                   "row": int(a["row"]), "col": int(a["col"]),
                   "raw_reports": int(a["raw_reports"]),
                   "nearest_prior_report_m": float(a["nearest_prior_report_m"]),
                   "new_site_at_250m": int(a["new_site_at_radius"]),
                   "own_cell_history_zero": int(a["own_cell_history_zero"]),
                   "raster_positive": int(a["raster_positive"]),
                   "nearby_history": float(a["nearby_history_selection_weight"]),
                   "old_cnn": float(old["model_selection_weight"]),
                   ARMS[0]: float(a["model_selection_weight"]),
                   ARMS[1]: float(b["model_selection_weight"]),
                   "source_selected_arm": selected_arm,
                   "source_selected": float((a if selected_arm == ARMS[0] else b)[
                       "model_selection_weight"])}
            all_rows.append(row)
        fold_summary.append({"fold": fold, "source_selected_arm": selected_arm,
                             "source_selected_alpha": source_validation[selected_arm][
                                 "source_selected_alpha"],
                             "new_events_250m": sum(r["new_site_at_250m"] for r in all_rows
                                                    if r["fold"] == fold),
                             "radii": radius_results})
        print("fold", fold, "selected", selected_arm, "events", len(one), flush=True)
    with SITES.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    new = [r for r in all_rows if r["new_site_at_250m"]]
    if len(all_rows) != 2326 or len(new) != 471 or sum(r["raster_positive"] for r in all_rows) != 2324:
        raise ValueError("Frozen raw-report site target or raster alignment changed")
    summary = {"all": summarize(all_rows, lambda r: True),
               "new_site_250m": summarize(all_rows, lambda r: r["new_site_at_250m"]),
               "250_to_750m": summarize(all_rows, lambda r: 250 < r["nearest_prior_report_m"] <= 750),
               "over_750m": summarize(all_rows, lambda r: r["nearest_prior_report_m"] > 750),
               "new_and_zero_own_cell_history": summarize(all_rows, lambda r:
                   r["new_site_at_250m"] and r["own_cell_history_zero"])}
    intervals = {}
    for comparator in ("nearby_history", "old_cnn"):
        intervals["selected_minus_"+comparator] = {
            "district_bootstrap_points": interval(new, "source_selected", comparator,
                                                   lambda r: r["district_index"], 804),
            "seven_day_bootstrap_points": interval(new, "source_selected", comparator,
                                                    lambda r: (int(r["day"][8:10])-1)//7+
                                                    (int(r["day"][5:7])-10)*5, 805)}
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "task": "First-in-prior-year reported physical-site event at a fixed daily district inspection budget",
              "source_selection": "Choose between two site-trained arms by Jan-Sep source-district 250m new-site capture; each arm's history blend was also selected there",
              "q4_2025": summary, "folds": fold_summary, "paired_intervals": intervals,
              "limits": ["Q4 2025 has been repeatedly inspected; model development and site radius are exploratory",
                         "Intervals reflect district or seven-day variation, not all selection uncertainty",
                         "A report-site medoid is not verified flood extent",
                         "Unreported cells are unlabeled, not observed dry"]}
    RESULT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"new_250m": summary["new_site_250m"],
                      "intervals": intervals}, indent=2), flush=True)


if __name__ == "__main__":
    main()
