"""Plot quota-free emerging-site capture by distance to prior reports.

All panels rank the same zero-own-history candidate pool per district/day.
Only the later reported-site outcome is split by nearest prior-report distance.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flood_multi_target_citywide_data import OUT


SOURCE = OUT / "q4_2025_site_novel_rank_curve_events.csv"
RESULT = OUT / "q4_2025_site_distance_rank_curves.json"
PLOT = OUT / "q4_2025_site_distance_rank_curves.png"
FRACTIONS = (.01, .02, .05, .10, .20, .30, .50)
BANDS = (
    ("All emerging sites", 250., float("inf")),
    ("250-500 m from an earlier report", 250., 500.),
    ("500-750 m from an earlier report", 500., 750.),
    (">750 m from an earlier report", 750., float("inf")),
)


def interval(rows, value, seed):
    by_district = defaultdict(list)
    for row in rows:
        by_district[int(row["district_index"])].append(row)
    districts = list(by_district)
    n = np.asarray([len(by_district[d]) for d in districts], np.float64)
    total = np.asarray([sum(value(r) for r in by_district[d]) for d in districts],
                       np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(districts), size=(5000, len(districts)))
    draws = total[sample].sum(axis=1)/n[sample].sum(axis=1)
    return [round(float(x), 3) for x in np.quantile(draws, (.025, .975))]


def main():
    with SOURCE.open(encoding="utf-8", newline="") as fh:
        events = list(csv.DictReader(fh))
    if len(events) != 393:
        raise ValueError("Expected 393 first-in-year zero-own-history site events")
    groups = []
    for i, (label, low, high) in enumerate(BANDS):
        rows = [r for r in events if low < float(r["nearest_prior_report_m"]) <= high]
        n = len(rows)
        if not n:
            raise ValueError(f"Empty distance group {label}")
        history_pct = np.mean([float(r["history_percentile"]) for r in rows])
        cnn_pct = np.mean([float(r["cnn_percentile"]) for r in rows])
        curve = []
        for f in FRACTIONS:
            h = sum(float(r[f"history_{f:g}"]) for r in rows)
            c = sum(float(r[f"cnn_{f:g}"]) for r in rows)
            curve.append({"fraction_of_unfamiliar_cells": f,
                          "history_hits": round(h, 4), "cnn_hits": round(c, 4),
                          "history_capture_pct": round(100*h/n, 3),
                          "cnn_capture_pct": round(100*c/n, 3),
                          "lift_points": round(100*(c-h)/n, 3)})
        groups.append({"label": label, "events": n,
                       "districts_with_events": len({int(r["district_index"]) for r in rows}),
                       "history_mean_rank_percentile": round(float(history_pct), 5),
                       "cnn_mean_rank_percentile": round(float(cnn_pct), 5),
                       "rank_lift_points": round(100*float(cnn_pct-history_pct), 3),
                       "rank_lift_district_bootstrap_95pct_points": interval(
                           rows, lambda r: 100*(float(r["cnn_percentile"])-
                                                float(r["history_percentile"])), 804+i),
                       "curve": curve})
    if sum(g["events"] for g in groups[1:]) != 393:
        raise ValueError("Disjoint distance bands do not cover all events")
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "task": "First-in-year reported-site capture among zero-own-history candidate cells, stratified by distance to nearest earlier report",
              "candidate_pool": "Same zero-own-history cells in every panel; only site outcomes are split by distance",
              "site_definition": "No earlier report within 250 m in the preceding 365 days",
              "groups": groups,
              "limits": ["The >750 m group has only 43 site events",
                         "District bootstrap intervals do not account for Q4 model-development reuse",
                         "Reported sites are not independently verified wet ground"]}
    RESULT.write_text(json.dumps(result, indent=2)+"\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.7), dpi=180,
                             sharex=True, sharey=True)
    for ax, group in zip(axes.flat, groups):
        curve = group["curve"]
        x = [0]+[100*r["fraction_of_unfamiliar_cells"] for r in curve]
        for key, color, name in (("history_capture_pct", "#8799a4", "Nearby history"),
                                 ("cnn_capture_pct", "#216d78", "Site CNN blend")):
            y = [0]+[r[key] for r in curve]
            ax.plot(x, y, color=color, lw=2.3, marker="o", markersize=3.5,
                    label=name)
        ax.set_title(f"{group['label']}  |  n={group['events']}\n"
                     f"Mean rank gain: {group['rank_lift_points']:+.1f} points",
                     fontsize=11.0, weight="bold")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 82)
        ax.set_xticks((0, 5, 10, 20, 30, 50))
        ax.set_yticks((0, 20, 40, 60, 80))
        ax.grid(alpha=.22)
        thirty = next(r for r in curve if r["fraction_of_unfamiliar_cells"] == .30)
        ax.text(.04, .91,
                f"At 30%: {thirty['history_capture_pct']:.1f}% history  →  "
                f"{thirty['cnn_capture_pct']:.1f}% CNN",
                transform=ax.transAxes, va="top", fontsize=8.6, color="#173044",
                bbox={"facecolor": "white", "edgecolor": "#d9e3e6",
                      "boxstyle": "round,pad=0.28"})
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.supxlabel("Top share of unfamiliar (zero-own-history) cells reviewed", y=.04)
    fig.supylabel("Share of emerging reported sites captured (%)", x=.02)
    fig.suptitle("Bangkok Q4 2025 | emerging sites by prior-report distance",
                 fontsize=13, y=.99)
    fig.text(.5, .01, "All panels rank the same unfamiliar candidate pool; distance splits the later site outcomes, not the cells being reviewed.",
             ha="center", fontsize=8.1, color="#4c5c69")
    fig.tight_layout(rect=(.025, .035, 1, .94), h_pad=2.2, w_pad=1.8)
    fig.savefig(PLOT, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"groups": [{"label": g["label"], "events": g["events"],
                                 "rank_lift_points": g["rank_lift_points"],
                                 "rank_ci": g["rank_lift_district_bootstrap_95pct_points"],
                                 "at_30pct": g["curve"][5]}
                                for g in groups]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
