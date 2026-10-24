"""Per-district Q4 novel-report capture versus available local report history.

The model is the saved five-fold target-masked CNN with ordinary-neighbor
propagation selected on source-district validation. District budgets are
separate, so the plotted percentages differ from pooled five-fold queues.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import diags
from scipy.stats import spearmanr

from flood_citywide_connected_score_smoother import messages
from flood_citywide_history_ranker import N_DAYS, TEST_START, load_grid
from flood_citywide_spatial_mask_augmentation import OUT_MODEL
from flood_citywide_spatial_novel_queue import (SpatialNovelCNN, collect,
                                                 queue_weights, static_image)
from flood_citywide_target_site_mask_test import q20_rows
from flood_citywide_terrain_continuity import links
from flood_multi_target_citywide_data import OUT


MODEL_RESULT = OUT / "q4_2025_mask_cnn_connected_score_smoother_fivefold.json"
CSV = OUT / "q4_2025_report_history_vs_novel_capture_by_district.csv"
JSON = OUT / "q4_2025_report_history_vs_novel_capture_by_district.json"
PLOT = OUT / "q4_2025_report_history_vs_novel_capture_by_district.png"
QUOTA = .20
ROLL = 9


def accumulate_district(data, arrays, score, ids):
    base, simple, _, novel, truth = arrays
    eligible = len(ids)
    result = {"positive": 0, "novel_positive": 0,
              "baseline_hits": 0., "model_hits": 0.,
              "baseline_novel_hits": 0., "model_novel_hits": 0.}
    for day in range(len(truth)):
        b, s, m = base[day, ids], simple[day, ids], score[day, ids]
        n, y = novel[day, ids], truth[day, ids]
        result["positive"] += int(y.sum())
        result["novel_positive"] += int(y[n].sum())
        w0 = queue_weights(b, s, n, QUOTA)
        w1 = queue_weights(b, m, n, QUOTA)
        if abs(float(w0.sum()) - .1 * eligible) > .02 or abs(float(w1.sum()) - .1 * eligible) > .02:
            raise ValueError("District inspection budget changed")
        result["baseline_hits"] += float(np.dot(w0, y))
        result["model_hits"] += float(np.dot(w1, y))
        result["baseline_novel_hits"] += float(np.dot(w0[n], y[n]))
        result["model_novel_hits"] += float(np.dot(w1[n], y[n]))
    return result


def group_capture(rows):
    positive = sum(r["q4_novel_positive_cell_days"] for r in rows)
    return {"districts": len(rows), "q4_novel_positive_cell_days": positive,
            "baseline_capture_pct": round(100 * sum(r["baseline_novel_hits"] for r in rows) / positive, 2),
            "model_capture_pct": round(100 * sum(r["model_novel_hits"] for r in rows) / positive, 2)}


def draw(rows, summary):
    ordered = sorted(rows, key=lambda r: (r["prior_365d_report_positive_cell_days"], r["district_code"]))
    x = np.asarray([r["prior_365d_report_positive_cell_days"] for r in ordered])
    n = np.asarray([r["q4_novel_positive_cell_days"] for r in ordered])
    y0 = np.asarray([r["baseline_novel_capture_pct"] for r in ordered])
    y1 = np.asarray([r["model_novel_capture_pct"] for r in ordered])
    fig, ax = plt.subplots(figsize=(12.8, 7.5), dpi=160)
    colors = {"baseline": "#087e83", "model": "#df5c2d"}
    size = 14 + 2.3 * n
    ax.scatter(x, y0, s=size, color=colors["baseline"], alpha=.24, edgecolors="none")
    ax.scatter(x, y1, s=size, color=colors["model"], alpha=.24, edgecolors="none")
    for key, label, color in (("baseline_novel_hits", "Nearby-history baseline", colors["baseline"]),
                              ("model_novel_hits", "Masked CNN + ordinary-neighbor paths", colors["model"])):
        centers, capture = [], []
        for start in range(len(ordered) - ROLL + 1):
            window = ordered[start:start+ROLL]
            centers.append(np.median([r["prior_365d_report_positive_cell_days"] for r in window]))
            capture.append(100 * sum(r[key] for r in window) /
                           sum(r["q4_novel_positive_cell_days"] for r in window))
        ax.plot(centers, capture, lw=2.6, color=color, label=label)
    lat = next((r for r in rows if r["district_code"] == "1011"), None)
    if lat is not None:
        ax.annotate("Lat Krabang", (lat["prior_365d_report_positive_cell_days"],
                                    lat["model_novel_capture_pct"]),
                    xytext=(9, 13), textcoords="offset points", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#555"})
    ax.set_xscale("log")
    ax.set_ylim(0, min(100, max(40, max(np.nanmax(y0), np.nanmax(y1)) + 7)))
    ax.set_xlabel("Report-positive cell-days in previous 365 days, by district")
    ax.set_ylabel("Q4 first-in-year report sites captured at daily top 10% cells (%)")
    ax.set_title("Does more local flood-report history improve novel-site ranking?", loc="left")
    ax.grid(axis="y", color="#d8dee3", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.text(.5, .01,
             "Dots: individual districts (size reflects Q4 positive count). Lines: rolling 9-district capture weighted by Q4 positives. "
             "Unreported cells are not verified dry.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(PLOT, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    data = load_grid()
    district = data["district_id"].ravel()[data["candidate_flat"]]
    history = (data["prefix"][TEST_START].ravel()[data["candidate_flat"]].astype(np.int64)
               - data["prefix"][TEST_START-365].ravel()[data["candidate_flat"]].astype(np.int64))
    ordinary, _, _, _, _, _ = links(data)
    degree = np.asarray(ordinary.sum(axis=1)).ravel()
    graph = diags(np.divide(1, degree, out=np.zeros_like(degree), where=degree > 0)) @ ordinary
    saved = torch.load(OUT_MODEL, map_location="cpu", weights_only=True)["states"]
    prior = json.loads(MODEL_RESULT.read_text())
    folds = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    rows, fold_audit = [], []
    for fold, members in enumerate(folds, 1):
        target = np.isin(district, members)
        static, _ = static_image(data, ~target)
        model = SpatialNovelCNN().eval()
        model.load_state_dict(saved[fold]["synthetic_center_block"])
        arrays = collect(data, model, static, range(TEST_START, N_DAYS))
        choice = prior["folds"][fold-1]["selected_on_source"]["ordinary"]
        selected = arrays[2]
        if choice["hops"]:
            for hops, message in messages(arrays[2], arrays[3], graph):
                if hops == choice["hops"]:
                    alpha = choice["alpha"]
                    selected = (1-alpha)*arrays[2] + alpha*message
                    break
        check = q20_rows(data, (arrays[0], arrays[1], selected, arrays[3], arrays[4]),
                         target, False)
        fold_audit.append({"fold": fold,
                           "q4_novel_positive": check["model"]["novel_positive_cell_days"],
                           "baseline_novel_capture": check["simple_neighbor"]["novel_capture"],
                           "model_novel_capture": check["model"]["novel_capture"]})
        for idx in members:
            ids = np.flatnonzero(district == idx)
            if not len(ids):
                continue
            out = accumulate_district(data, arrays, selected, ids)
            if not out["novel_positive"]:
                continue
            row = {"district_code": data["districts"][idx]["code"],
                   "district": data["districts"][idx]["name_en"],
                   "fold": fold, "candidate_cells": len(ids),
                   "prior_365d_report_positive_cell_days": int(history[ids].sum()),
                   "prior_365d_positive_days_per_100_cells": round(float(
                       100 * history[ids].sum()/len(ids)), 2),
                   "q4_report_positive_cell_days": out["positive"],
                   "q4_novel_positive_cell_days": out["novel_positive"],
                   "baseline_novel_hits": round(out["baseline_novel_hits"], 5),
                   "model_novel_hits": round(out["model_novel_hits"], 5),
                   "baseline_novel_capture_pct": round(
                       100 * out["baseline_novel_hits"]/out["novel_positive"], 3),
                   "model_novel_capture_pct": round(
                       100 * out["model_novel_hits"]/out["novel_positive"], 3),
                   "model_minus_baseline_points": round(100 * (
                       out["model_novel_hits"]-out["baseline_novel_hits"])/out["novel_positive"], 3)}
            rows.append(row)
        print("fold", fold, "districts", len(members), "model capture", check["model"]["novel_capture"], flush=True)
    if not rows:
        raise ValueError("No Q4 novel report sites")
    # Confirm that the same saved scores and fold-wide queue reproduce the
    # previously published ~14% result before changing to district budgets.
    fold_novel = sum(x["q4_novel_positive"] for x in fold_audit)
    fold_model_hits = sum(x["q4_novel_positive"]*x["model_novel_capture"] for x in fold_audit)
    fold_baseline_hits = sum(x["q4_novel_positive"]*x["baseline_novel_capture"] for x in fold_audit)
    if fold_novel != 1664 or abs(fold_model_hits/fold_novel-.14263) > .00002:
        raise ValueError("Saved model/fold queue does not match earlier 14.263%")
    rows.sort(key=lambda r: (r["prior_365d_report_positive_cell_days"], r["district_code"]))
    with CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    half = len(rows)//2
    lower, upper = group_capture(rows[:half]), group_capture(rows[half:])
    def pooled_lift(group):
        return 100 * sum(r["model_novel_hits"]-r["baseline_novel_hits"]
                         for r in group) / sum(r["q4_novel_positive_cell_days"]
                                               for r in group)
    rng = np.random.default_rng(2402)
    high_minus_low = []
    for _ in range(5000):
        low_sample = [rows[i] for i in rng.integers(half, size=half)]
        high_sample = [rows[half+i] for i in rng.integers(len(rows)-half,
                                                          size=len(rows)-half)]
        high_minus_low.append(pooled_lift(high_sample)-pooled_lift(low_sample))
    x = np.asarray([r["prior_365d_report_positive_cell_days"] for r in rows])
    density = np.asarray([r["prior_365d_positive_days_per_100_cells"] for r in rows])
    district_lift = np.asarray([r["model_minus_baseline_points"] for r in rows])
    summary = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "unit": "district", "districts_with_q4_novel_positives": len(rows),
               "x": "Pre-Q4 365-day report-positive cell-days in district; repeated complaints in one cell-day are collapsed",
               "y": "Share of Q4 first-in-prior-year report-positive cell-days in a daily top-10% district cell queue with 20% novel slot reserve",
               "baseline": "Fixed nearby-history novel score, same queue budget",
               "model": "Saved geographically held-out masked CNN + source-selected ordinary-neighbor propagation",
               "pooled_fold_queue_audit": {"novel_positives": fold_novel,
                   "baseline_capture": round(fold_baseline_hits/fold_novel, 5),
                   "model_capture": round(fold_model_hits/fold_novel, 5)},
               "district_budget_pooled": group_capture(rows),
               "low_history_half": lower, "high_history_half": upper,
               "high_minus_low_model_lift_points": round(
                   pooled_lift(rows[half:])-pooled_lift(rows[:half]), 3),
               "high_minus_low_district_bootstrap_95pct_interval_points": np.quantile(
                   high_minus_low, [.025, .975]).round(3).tolist(),
               "spearman_history_count_vs_model_lift": round(float(spearmanr(x, district_lift).statistic), 4),
               "spearman_history_density_vs_model_lift": round(float(spearmanr(density, district_lift).statistic), 4),
               "median_q4_novel_positives_per_district": float(np.median([
                   r["q4_novel_positive_cell_days"] for r in rows])),
               "note": "Local history count measures context available in the held-out district, not examples used to fit CNN weights. Per-district budgets yield different percentages from pooled fold budgets. Q4 was examined in prior studies; this is exploratory.",
               "rows": rows}
    JSON.write_text(json.dumps(summary, indent=2) + "\n")
    draw(rows, summary)
    print(json.dumps({k: summary[k] for k in (
        "pooled_fold_queue_audit", "district_budget_pooled", "low_history_half",
        "high_history_half", "high_minus_low_model_lift_points",
        "high_minus_low_district_bootstrap_95pct_interval_points",
        "spearman_history_count_vs_model_lift",
        "spearman_history_density_vs_model_lift")}, indent=2))
    print("saved", PLOT, CSV, JSON)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--plot-only":
        saved = json.loads(JSON.read_text())
        draw(saved["rows"], saved)
        print("saved", PLOT)
    else:
        main()
