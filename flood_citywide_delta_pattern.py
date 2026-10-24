"""Explain district-level lift of the saved masked CNN + ordinary paths.

Q4 2025 first-in-prior-year Traffy positives only. Replays each district's
daily 10% queue, then tags positive cells by pre-day history reach (1/3
side-neighbor hops), built/crop cover, and district sample size.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.sparse import diags
from scipy.stats import spearmanr

from flood_citywide_connected_score_smoother import messages
from flood_citywide_data_count_capture_curve import JSON as PREVIOUS
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, load_grid
from flood_citywide_spatial_mask_augmentation import OUT_MODEL
from flood_citywide_spatial_novel_queue import (SpatialNovelCNN, collect,
                                                 queue_weights, static_image)
from flood_citywide_terrain_continuity import links
from flood_multi_target_citywide_data import OUT
from flood_multi_target_data import START
from flood_open_weak_data import FEATURE_NAMES


MODEL_RESULT = OUT / "q4_2025_mask_cnn_connected_score_smoother_fivefold.json"
RESULT = OUT / "q4_2025_novel_delta_pattern.json"
DISTRICTS = OUT / "q4_2025_novel_delta_pattern_districts.csv"
SITES = OUT / "q4_2025_novel_delta_pattern_positive_sites.csv"
PLOT = OUT / "q4_2025_novel_delta_pattern.png"
PLOT_DENSITY = OUT / "q4_2025_novel_delta_vs_report_density.png"
QUOTA = .20
STRATA = ("adjacent", "two_to_three_hops", "beyond_three_hops")
CROSS = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def site_summary(rows):
    out = {}
    for name in STRATA:
        group = [r for r in rows if r["prior_report_reach"] == name]
        count = len(group)
        out[name] = {"positive_cell_days": count,
                     "baseline_captured": round(sum(r["baseline_selection_weight"] for r in group), 3),
                     "model_captured": round(sum(r["model_selection_weight"] for r in group), 3),
                     "baseline_capture_pct": round(100 * sum(r["baseline_selection_weight"] for r in group) / count, 2)
                     if count else None,
                     "model_capture_pct": round(100 * sum(r["model_selection_weight"] for r in group) / count, 2)
                     if count else None}
        out[name]["lift_points"] = round(out[name]["model_capture_pct"]-
                                          out[name]["baseline_capture_pct"], 2) if count else None
    return out


def draw(rows, strata):
    eligible = [r for r in rows if r["q4_novel_positive_cell_days"] >= 20]
    ordered = sorted(eligible, key=lambda r: r["model_minus_baseline_points"])
    bottom, top = ordered[:8], ordered[-8:]
    shown = bottom + top
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.5, 7), dpi=160,
                                gridspec_kw={"width_ratios": [1.45, 1]},
                                constrained_layout=True)
    y = np.arange(len(shown))
    delta = [r["model_minus_baseline_points"] for r in shown]
    ax.barh(y, delta, color=["#c45742" if v < 0 else "#138b82" for v in delta])
    ax.set_yticks(y, [r["district"] for r in shown], fontsize=8)
    ax.set_xlabel("Model minus nearby-history capture (percentage points)")
    ax.axvline(0, color="#343b43", lw=1)
    ax.set_title("Largest district gains and losses\n(≥20 new Q4 report cell-days)", loc="left")
    ax.grid(axis="x", color="#dee2e5", lw=.6)
    for j, row in enumerate(shown):
        value = row["model_minus_baseline_points"]
        ax.text(value + (.35 if value >= 0 else -.35), j,
                f"{row['q4_novel_positive_cell_days']} sites",
                ha="left" if value >= 0 else "right", va="center", fontsize=7)
    categories = ["Adjacent", "2–3 hops", ">3 hops"]
    values0 = [strata[k]["baseline_capture_pct"] for k in STRATA]
    values1 = [strata[k]["model_capture_pct"] for k in STRATA]
    pos = np.arange(3)
    bx.bar(pos-.18, values0, width=.35, color="#087e83", label="Nearby history")
    bx.bar(pos+.18, values1, width=.35, color="#df5c2d", label="CNN + paths")
    bx.set_xticks(pos, categories)
    bx.set_ylim(0, max(values0+values1)*1.25)
    bx.set_ylabel("New report sites captured at top-10% district budget (%)")
    bx.set_title("Does proximity to earlier reports explain lift?", loc="left")
    bx.legend(frameon=False)
    bx.grid(axis="y", color="#dee2e5", lw=.6)
    for j, key in enumerate(STRATA):
        bx.text(j, max(values0[j], values1[j])+.4,
                f"n={strata[key]['positive_cell_days']}", ha="center", fontsize=8)
    fig.savefig(PLOT, bbox_inches="tight")
    plt.close(fig)


def draw_density(rows):
    ordered = sorted(rows, key=lambda r: r["prior_365d_positive_days_per_100_cells"])
    x = np.asarray([r["prior_365d_positive_days_per_100_cells"] for r in ordered])
    y = np.asarray([r["model_minus_baseline_points"] for r in ordered])
    n = np.asarray([r["q4_novel_positive_cell_days"] for r in ordered])
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=160)
    ax.scatter(x, y, s=20+2.3*n, c=np.where(y >= 0, "#138b82", "#c45742"),
               alpha=.62, edgecolor="white", linewidth=.5)
    centers, rolling = [], []
    for start in range(len(ordered)-9+1):
        group = ordered[start:start+9]
        positives = sum(r["q4_novel_positive_cell_days"] for r in group)
        centers.append(np.median([r["prior_365d_positive_days_per_100_cells"]
                                  for r in group]))
        rolling.append(100*sum(r["net_extra_captured_sites"] for r in group)/positives)
    ax.plot(centers, rolling, color="#283a4b", lw=2.4,
            label="Rolling 9-district, positive-weighted gain")
    ax.axhline(0, color="#555", lw=.9)
    ax.axvline((ordered[24]["prior_365d_positive_days_per_100_cells"]+
                ordered[25]["prior_365d_positive_days_per_100_cells"])/2,
               color="#777", lw=.8, ls="--", label="Half-split: 78 per 100 cells")
    for code in ("1021", "1011", "1025", "1022"):
        row = next((r for r in rows if r["district_code"] == code), None)
        if row is not None:
            ax.annotate(row["district"],
                        (row["prior_365d_positive_days_per_100_cells"],
                         row["model_minus_baseline_points"]),
                        xytext=(7, 7), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Prior-year report-positive days per 100 district cells")
    ax.set_ylabel("Model minus nearby-history capture (percentage points)")
    ax.set_title("The model's advantage tends to shrink as local report density rises",
                 loc="left")
    ax.grid(color="#e3e7ea", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.text(.5, .01, "Dots are districts; size reflects Q4 new-site count. Descriptive Q4 analysis, not a causal data-size experiment.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(PLOT_DENSITY, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    data = load_grid()
    prior = json.loads(MODEL_RESULT.read_text())
    previous = json.loads(PREVIOUS.read_text())
    district_id = data["district_id"].ravel()[data["candidate_flat"]]
    flat = data["candidate_flat"]
    rr, cc = np.unravel_index(flat, SHAPE)
    built = data["static"][:, FEATURE_NAMES.index("worldcover_built_fraction_2021")]
    crop = data["static"][:, FEATURE_NAMES.index("worldcover_crop_fraction_2021")]
    ordinary, _, _, _, _, _ = links(data)
    degree = np.asarray(ordinary.sum(axis=1)).ravel()
    graph = diags(np.divide(1, degree, out=np.zeros_like(degree), where=degree > 0)) @ ordinary
    saved = torch.load(OUT_MODEL, map_location="cpu", weights_only=True)["states"]
    folds = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    sites = []
    for fold, members in enumerate(folds, 1):
        target = np.isin(district_id, members)
        static, _ = static_image(data, ~target)
        model = SpatialNovelCNN().eval()
        model.load_state_dict(saved[fold]["synthetic_center_block"])
        arrays = collect(data, model, static, range(TEST_START, N_DAYS))
        choice = prior["folds"][fold-1]["selected_on_source"]["ordinary"]
        score = arrays[2]
        if choice["hops"]:
            for hops, message in messages(arrays[2], arrays[3], graph):
                if hops == choice["hops"]:
                    score = ((1-choice["alpha"])*score + choice["alpha"]*message)
                    break
        for day in range(len(arrays[4])):
            absolute_day = TEST_START + day
            reports_before = (data["prefix"][absolute_day]
                              - data["prefix"][absolute_day-365]) > 0
            adjacent = binary_dilation(reports_before, structure=CROSS, iterations=1).ravel()[flat]
            within3 = binary_dilation(reports_before, structure=CROSS, iterations=3).ravel()[flat]
            for idx in members:
                ids = np.flatnonzero(district_id == idx)
                b, s, m = arrays[0][day, ids], arrays[1][day, ids], score[day, ids]
                novel, truth = arrays[3][day, ids], arrays[4][day, ids]
                positive = np.flatnonzero(novel & truth.astype(bool))
                if not len(positive):
                    continue
                w0 = queue_weights(b, s, novel, QUOTA)
                w1 = queue_weights(b, m, novel, QUOTA)
                for j in positive:
                    cell = ids[j]
                    reach = (STRATA[0] if adjacent[cell] else
                             STRATA[1] if within3[cell] else STRATA[2])
                    sites.append({"day": str(START + timedelta(days=absolute_day)),
                                  "district_code": data["districts"][idx]["code"],
                                  "district": data["districts"][idx]["name_en"],
                                  "row": int(rr[cell]), "col": int(cc[cell]),
                                  "prior_report_reach": reach,
                                  "built_fraction": round(float(built[cell]), 5),
                                  "crop_fraction": round(float(crop[cell]), 5),
                                  "baseline_selection_weight": round(float(w0[j]), 6),
                                  "model_selection_weight": round(float(w1[j]), 6),
                                  "selection_delta": round(float(w1[j]-w0[j]), 6)})
        print("fold", fold, "positive sites accumulated", len(sites), flush=True)
    if len(sites) != 1664:
        raise ValueError("Novel positive cell-day count changed")
    write_csv(SITES, sites)
    by_district = defaultdict(list)
    for row in sites:
        by_district[row["district_code"]].append(row)
    rows = []
    for previous_row in previous["rows"]:
        group = by_district[previous_row["district_code"]]
        if len(group) != previous_row["q4_novel_positive_cell_days"]:
            raise ValueError("District positive count changed")
        baseline = sum(r["baseline_selection_weight"] for r in group)
        model = sum(r["model_selection_weight"] for r in group)
        if (abs(baseline-previous_row["baseline_novel_hits"]) > .01 or
                abs(model-previous_row["model_novel_hits"]) > .01):
            raise ValueError("District queue no longer matches two-line curve")
        row = dict(previous_row)
        for name in STRATA:
            row[name+"_positive"] = sum(r["prior_report_reach"] == name for r in group)
            row[name+"_share"] = round(row[name+"_positive"]/len(group), 4)
            row[name+"_lift_points"] = round(100*sum(r["selection_delta"] for r in group
                if r["prior_report_reach"] == name)/max(1, row[name+"_positive"]), 3)
        row["positive_mean_built_fraction"] = round(float(np.mean([r["built_fraction"] for r in group])), 4)
        row["positive_mean_crop_fraction"] = round(float(np.mean([r["crop_fraction"] for r in group])), 4)
        row["net_extra_captured_sites"] = round(model-baseline, 3)
        rows.append(row)
    write_csv(DISTRICTS, rows)
    strata = site_summary(sites)
    def corr(field):
        return round(float(spearmanr([r[field] for r in rows],
                                     [r["model_minus_baseline_points"] for r in rows]).statistic), 4)
    features = {key: corr(key) for key in (
        "candidate_cells", "prior_365d_report_positive_cell_days",
        "prior_365d_positive_days_per_100_cells", "q4_novel_positive_cell_days",
        "adjacent_share", "two_to_three_hops_share", "beyond_three_hops_share",
        "positive_mean_built_fraction", "positive_mean_crop_fraction")}
    strong = [r for r in rows if r["q4_novel_positive_cell_days"] >= 20]
    density_order = sorted(rows, key=lambda r: r["prior_365d_positive_days_per_100_cells"])
    low_density, high_density = density_order[:25], density_order[25:]
    def group_lift(group):
        positives = sum(r["q4_novel_positive_cell_days"] for r in group)
        baseline = sum(r["baseline_novel_hits"] for r in group)
        model = sum(r["model_novel_hits"] for r in group)
        return {"districts": len(group), "novel_positive_cell_days": positives,
                "baseline_capture_pct": round(100*baseline/positives, 2),
                "model_capture_pct": round(100*model/positives, 2),
                "lift_points": round(100*(model-baseline)/positives, 2)}
    rng = np.random.default_rng(255)
    density_differences = []
    for _ in range(5000):
        low_sample = [low_density[i] for i in rng.integers(25, size=25)]
        high_sample = [high_density[i] for i in rng.integers(25, size=25)]
        density_differences.append(group_lift(low_sample)["lift_points"]-
                                   group_lift(high_sample)["lift_points"])
    without_bang_khun_thian = [r for r in low_density if r["district_code"] != "1021"]
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "task": "Why does CNN + ordinary paths gain in some district report queues but lose in others?",
              "outcome": "Q4 2025 first-in-prior-year report-positive cell-days at separate daily top-10% district budgets",
              "pre_day_reach": "Shortest Manhattan reach to a cell with any report in preceding 365 days; 1, 2-3, or >3 side-neighbor 250 m hops. Reach is spatial history, not proven water flow.",
              "total_positive_cell_days": len(sites),
              "districts_model_better": sum(r["net_extra_captured_sites"] > .01 for r in rows),
              "districts_baseline_better": sum(r["net_extra_captured_sites"] < -.01 for r in rows),
              "districts_tied": sum(abs(r["net_extra_captured_sites"]) <= .01 for r in rows),
              "strata": strata, "district_spearman_with_lift_points": features,
              "report_density_halves": {
                  "split_positive_days_per_100_cells": [
                      low_density[-1]["prior_365d_positive_days_per_100_cells"],
                      high_density[0]["prior_365d_positive_days_per_100_cells"]],
                  "lower": group_lift(low_density), "upper": group_lift(high_density),
                  "lower_minus_upper_lift_points_district_bootstrap_95pct": np.quantile(
                      density_differences, [.025, .975]).round(2).tolist(),
                  "lower_without_bang_khun_thian": group_lift(without_bang_khun_thian)},
              "top_gains_at_least_20_positives": sorted(strong, key=lambda r: -r["net_extra_captured_sites"])[:8],
              "top_losses_at_least_20_positives": sorted(strong, key=lambda r: r["net_extra_captured_sites"])[:8],
              "note": "Descriptive associations only; sites are clustered across days and nearby cells, Q4 has been examined repeatedly, and report absence is not dry.",
              "districts": rows}
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    draw(rows, strata)
    draw_density(rows)
    print(json.dumps({key: result[key] for key in (
        "districts_model_better", "districts_baseline_better", "districts_tied",
        "strata", "district_spearman_with_lift_points", "report_density_halves")}, indent=2))
    print("saved", RESULT, DISTRICTS, SITES, PLOT, PLOT_DENSITY)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--plot-only":
        saved = json.loads(RESULT.read_text())
        draw(saved["districts"], saved["strata"])
        draw_density(saved["districts"])
        print("saved", PLOT, PLOT_DENSITY)
    else:
        main()
