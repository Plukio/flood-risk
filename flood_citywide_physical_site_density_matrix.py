"""Visualize site-trained CNN gain by prior report density and distance.

Uses the frozen, source-selected five-fold models. Density is fixed before
Q4 2025: prior 365-day report-positive cell-days per 100 district cells.
Site grouping stays at 250 m; distance thresholds create nested report-site
sets. This is not a measure of verified flood extent.
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
import torch
from matplotlib.colors import TwoSlopeNorm

from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, load_grid
from flood_citywide_physical_site_data import RADII, build
from flood_citywide_physical_site_final import RESULT as FINAL_RESULT
from flood_citywide_physical_site_replay import event_capture, group_events
from flood_citywide_physical_site_residual import MODEL, score_days
from flood_citywide_spatial_novel_queue import SpatialNovelCNN, static_image
from flood_multi_target_citywide_data import OUT


DENSITY_SOURCE = OUT / "q4_2025_report_history_vs_novel_capture_by_district.csv"
PNG = OUT / "q4_2025_physical_site_density_distance_matrix.png"
DISTRICT_PNG = OUT / "q4_2025_physical_site_250m_lift_by_district_density.png"
CSV = OUT / "q4_2025_physical_site_density_distance_matrix.csv"
DISTRICT_CSV = OUT / "q4_2025_physical_site_250m_lift_by_district_density.csv"
EVENTS = OUT / "q4_2025_physical_site_density_distance_events.csv"
JSON = OUT / "q4_2025_physical_site_density_distance_matrix.json"
GROUPS = ("Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def density_groups(districts):
    with DENSITY_SOURCE.open(encoding="utf-8", newline="") as file:
        density = {r["district_code"]: float(
            r["prior_365d_positive_days_per_100_cells"]) for r in csv.DictReader(file)}
    if len(density) != len(districts):
        raise ValueError("District density source no longer covers all districts")
    ordered = sorted(range(len(districts)), key=lambda i: (density[districts[i]["code"]],
                                                            districts[i]["code"]))
    bins = [ordered[i:i+10] for i in range(0, 50, 10)]
    group = {i: GROUPS[j] for j, members in enumerate(bins) for i in members}
    detail = {GROUPS[j]: {
        "districts": len(members),
        "min_positive_cell_days_per_100_cells": min(density[districts[i]["code"]]
                                                    for i in members),
        "max_positive_cell_days_per_100_cells": max(density[districts[i]["code"]]
                                                    for i in members)}
        for j, members in enumerate(bins)}
    return group, density, detail


def interval(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["district_code"]].append(row)
    districts = list(groups)
    n = np.asarray([len(groups[d]) for d in districts], np.float64)
    delta = np.asarray([sum(r["model_weight"]-r["baseline_weight"]
                            for r in groups[d]) for d in districts], np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(districts), size=(5000, len(districts)))
    draws = 100*delta[sample].sum(axis=1)/n[sample].sum(axis=1)
    return [round(float(x), 2) for x in np.quantile(draws, [.025, .975])]


def summarize(events):
    rows = []
    for radius in RADII:
        for group in GROUPS:
            selected = [r for r in events if r["radius_m"] == radius and
                        r["density_group"] == group]
            n = len(selected)
            base = sum(r["baseline_weight"] for r in selected)
            model = sum(r["model_weight"] for r in selected)
            rows.append({"radius_m": radius, "density_group": group,
                         "districts_in_group": 10,
                         "districts_with_new_sites": len({r["district_code"] for r in selected}),
                         "new_site_events": n,
                         "zero_own_history_events": sum(r["own_cell_history_zero"] for r in selected),
                         "baseline_hits": round(base, 5),
                         "model_hits": round(model, 5),
                         "baseline_capture_pct": round(100*base/n, 3) if n else None,
                         "model_capture_pct": round(100*model/n, 3) if n else None,
                         "lift_points": round(100*(model-base)/n, 3) if n else None,
                         "model_over_baseline": round(model/base, 3) if base else None,
                         "district_bootstrap_lift_95pct_points": interval(selected,
                             37000+radius+GROUPS.index(group)) if n else None})
    return rows


def draw(rows, detail):
    def get(radius, group):
        return next(r for r in rows if r["radius_m"] == radius and
                    r["density_group"] == group)
    delta = np.asarray([[get(r, g)["lift_points"] for g in GROUPS]
                        for r in RADII], np.float64)
    maximum = max(5., float(np.nanmax(np.abs(delta))))
    fig, ax = plt.subplots(figsize=(16.2, 8.4), dpi=180)
    fig.patch.set_facecolor("#f8fafb")
    ax.set_facecolor("#f8fafb")
    palette = plt.get_cmap("RdBu").copy()
    palette.set_bad("#e6e9ec")
    image = ax.imshow(delta, cmap=palette, norm=TwoSlopeNorm(vmin=-maximum,
                      vcenter=0, vmax=maximum), aspect="auto")
    ax.set_xticks(range(len(GROUPS)), [
        f"{g}\n{detail[g]['min_positive_cell_days_per_100_cells']:.0f}–"
        f"{detail[g]['max_positive_cell_days_per_100_cells']:.0f}"
        for g in GROUPS], fontsize=10.5)
    ax.set_yticks(range(3), [f"No prior report within {r} m" for r in RADII],
                  fontsize=11)
    ax.set_xlabel("Prior-year report-positive days per 100 district cells (10 districts per band)",
                  fontsize=12, labelpad=14)
    ax.set_ylabel("No earlier report within distance\n(previous 365 days)",
                  fontsize=12, labelpad=14)
    ax.set_xticks(np.arange(-.5, len(GROUPS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    for i, radius in enumerate(RADII):
        for j, group in enumerate(GROUPS):
            row = get(radius, group)
            if not row["new_site_events"]:
                ax.text(j, i, "No sites", ha="center", va="center",
                        color="#596875", fontsize=11)
                continue
            foreground = "white" if abs(delta[i, j]) > .58*maximum else "#173044"
            base = row["baseline_capture_pct"]
            model = row["model_capture_pct"]
            lift = row["lift_points"]
            ratio = row["model_over_baseline"]
            ratio_text = f" · {ratio:.1f}×" if ratio is not None and row["baseline_hits"] >= 2 else ""
            lift_text = "0.0 pp" if abs(lift) < .05 else f"{lift:+.1f} pp"
            ax.text(j, i-.23, lift_text, ha="center", va="center",
                    color=foreground, fontsize=15, fontweight="bold")
            ax.text(j, i+.03, f"{base:.1f}% → {model:.1f}%{ratio_text}",
                    ha="center", va="center", color=foreground, fontsize=9.5)
            size = f"n={row['new_site_events']}"
            if row["new_site_events"] < 20:
                size += " · small"
            ax.text(j, i+.30, size, ha="center", va="center", color=foreground,
                    fontsize=9.5)
    bar = fig.colorbar(image, ax=ax, shrink=.75, pad=.025)
    bar.set_label("Site-trained CNN minus nearby history (percentage points)",
                  fontsize=10)
    fig.suptitle("Site-trained CNN lift by report density and distance",
                 x=.06, ha="left", fontsize=18, fontweight="bold", color="#173044")
    fig.text(.06, .91,
             "Q4 2025 reported-site capture at the same daily top-10% district cell budget",
             ha="left", fontsize=11, color="#516575")
    fig.text(.06, .035,
             "All rows use the same 250 m site grouping; distance thresholds form nested site sets. Capture is of reports, not verified flooding. "
             "A ratio is omitted where the baseline has fewer than 2 expected hits.",
             ha="left", fontsize=8.8, color="#516575")
    fig.subplots_adjust(left=.22, right=.94, top=.87, bottom=.19)
    fig.savefig(PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def district_detail(site_rows):
    with DENSITY_SOURCE.open(encoding="utf-8", newline="") as file:
        names = {r["district_code"]: r["district"] for r in csv.DictReader(file)}
    grouped = defaultdict(list)
    for row in site_rows:
        if row["radius_m"] == 250:
            grouped[row["district_code"]].append(row)
    out = []
    for code, events in grouped.items():
        n = len(events)
        base = sum(r["baseline_weight"] for r in events)
        model = sum(r["model_weight"] for r in events)
        out.append({"district_code": code, "district": names[code],
                    "density_group": events[0]["density_group"],
                    "pre_q4_report_days_per_100_cells": events[0][
                        "pre_q4_report_days_per_100_cells"],
                    "new_site_events": n,
                    "nearby_history_hits": round(base, 5),
                    "site_cnn_hits": round(model, 5),
                    "lift_points": round(100*(model-base)/n, 3),
                    "extra_selected_events": round(model-base, 5)})
    return sorted(out, key=lambda r: r["pre_q4_report_days_per_100_cells"])


def draw_district(rows):
    fig, ax = plt.subplots(figsize=(12.5, 6.7), dpi=180)
    fig.patch.set_facecolor("#f8fafb")
    ax.set_facecolor("#f8fafb")
    for row in rows:
        gain = row["lift_points"]
        color = "#196f9d" if gain > .05 else "#bd6a51" if gain < -.05 else "#8b969e"
        ax.scatter(row["pre_q4_report_days_per_100_cells"], gain,
                   s=24+7*row["new_site_events"], color=color,
                   alpha=.72, edgecolor="white", linewidth=.7)
    ax.axhline(0, color="#53616c", linewidth=1)
    overall = sum(r["extra_selected_events"] for r in rows)/sum(
        r["new_site_events"] for r in rows)*100
    ax.axhline(overall, color="#196f9d", linewidth=1.3, linestyle="--",
               label=f"All districts: {overall:+.1f} pp")
    prominent = sorted((r for r in rows if r["new_site_events"] >= 8),
                       key=lambda r: abs(r["extra_selected_events"]), reverse=True)[:5]
    for row in prominent:
        ax.annotate(row["district"],
                    (row["pre_q4_report_days_per_100_cells"], row["lift_points"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8,
                    color="#344b5b")
    ax.set_xlabel("Prior-year report-positive days per 100 district cells", fontsize=11)
    ax.set_ylabel("CNN minus nearby-history capture (percentage points)", fontsize=11)
    ax.set_title("250 m new-site lift in each Bangkok district", loc="left",
                 fontsize=16, fontweight="bold", color="#173044")
    ax.grid(color="#dce3e8", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.text(.5, .02, "Each dot is a district; dot area grows with its Q4 new-site count. "
             "Large lifts from very few sites are unstable.", ha="center",
             fontsize=8.5, color="#516575")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(DISTRICT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    data = load_grid()
    event_set = build(data["candidate"], data["district_id"])[250]
    group, density, detail = density_groups(data["districts"])
    final = json.loads(FINAL_RESULT.read_text())
    saved = torch.load(MODEL, map_location="cpu", weights_only=True)["states"]
    flat = data["candidate_flat"]
    lookup = np.full(np.prod(SHAPE), -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    district_id = data["district_id"].ravel()[flat]
    folds = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    site_rows = []
    for fold, members in enumerate(folds, 1):
        selected = final["folds"][fold-1]
        arm, alpha = selected["source_selected_arm"], selected["source_selected_alpha"]
        model = SpatialNovelCNN().eval()
        model.load_state_dict(saved[fold][arm])
        static, _ = static_image(data, ~np.isin(district_id, members))
        arrays, score = score_days(data, model, static, range(TEST_START, N_DAYS))
        allowed = np.zeros(len(data["districts"]), bool)
        allowed[members] = True
        grouped = group_events(event_set, allowed, TEST_START, N_DAYS, lookup)
        result, rows = event_capture(arrays, score, grouped, district_id,
                                      members, 250, alpha, event_rows=True)
        for row in rows:
            for radius in RADII:
                if float(row["nearest_prior_report_m"]) <= radius:
                    continue
                district = int(row["district_index"])
                code = data["districts"][district]["code"]
                site_rows.append({"radius_m": radius, "day": row["day"],
                                  "district_code": code,
                                  "density_group": group[district],
                                  "pre_q4_report_days_per_100_cells": density[code],
                                  "row": row["row"], "col": row["col"],
                                  "nearest_prior_report_m": row["nearest_prior_report_m"],
                                  "own_cell_history_zero": row["own_cell_history_zero"],
                                  "baseline_weight": float(row["nearby_history_selection_weight"]),
                                  "model_weight": float(row["model_selection_weight"])})
        expected = selected["radii"][arm]["250"]["new_site"]
        if result["new_site"]["events"] != expected["events"] or abs(
                result["new_site"]["model_hits"]-expected["model_hits"]) > .0001:
            raise ValueError("250 m replay differs from frozen final model")
        print("fold", fold, "complete", flush=True)
    rows = summarize(site_rows)
    by_radius = {}
    for radius in RADII:
        same = [r for r in rows if r["radius_m"] == radius]
        by_radius[str(radius)] = {"events": sum(r["new_site_events"] for r in same),
                                  "baseline_hits": round(sum(r["baseline_hits"] for r in same), 4),
                                  "model_hits": round(sum(r["model_hits"] for r in same), 4)}
    if by_radius["250"]["events"] != 471 or abs(
            by_radius["250"]["model_hits"]-47.1567) > .01 or abs(
            by_radius["250"]["baseline_hits"]-26.74) > .01:
        raise ValueError("Density matrix does not reproduce frozen 250 m result")
    write_csv(EVENTS, site_rows)
    write_csv(CSV, rows)
    draw(rows, detail)
    districts = district_detail(site_rows)
    write_csv(DISTRICT_CSV, districts)
    draw_district(districts)
    output = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "axes": {"x": "Pre-Q4 365-day district report-positive cell-days per 100 candidate cells, quintiles of 50 districts",
                       "y": "Nearest raw direct Traffy flood report from an earlier day in the prior 365 days exceeds the threshold; same 250m site clusters for every row"},
              "density_groups": detail, "same_queue": "Top 10% of cells per district per day; 20% of slots reserved for zero-own-365-day-history cells",
              "model": "Source-selected site-trained CNN arm and blend; no Q4 arm selection",
              "radius_totals": by_radius, "matrix": rows,
              "district_detail": districts,
              "limits": ["Q4 2025 and the density relationship were previously explored",
                         "Strict-distance cells may have very few reported new sites, so ratios are unstable",
                         "Rows are nested thresholds over the same 250m site groups",
                         "Report locations are not verified flood extent; missing reports are not dry labels"]}
    JSON.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
