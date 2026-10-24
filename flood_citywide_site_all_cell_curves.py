"""Full district-cell capture curves for all and emerging reported sites.

No discovery quota is imposed. The full queue orders every candidate cell by
the recent-plus-nearby history score. The CNN arm adds its trained residual to
zero-own-history cells only, with a nonnegative coefficient chosen on Jan-Sep
2025 source-district new-site mean rank. Q4 excluded districts are evaluated.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, VALID_START, load_grid
from flood_citywide_physical_site_data import build
from flood_citywide_physical_site_replay import group_events, pct
from flood_citywide_physical_site_residual import MODEL, score_days
from flood_citywide_spatial_novel_queue import SpatialNovelCNN, static_image, top_weights
from flood_multi_target_citywide_data import OUT


RANK = OUT / "q4_2025_site_novel_rank_curve.json"
RESULT = OUT / "q4_2025_site_all_cell_curves.json"
EVENTS = OUT / "q4_2025_site_all_cell_curves_events.csv"
PLOT = OUT / "q4_2025_site_all_cell_curves.png"
FRACTIONS = (.01, .02, .05, .10, .15, .20, .30, .50)
LAMBDA_GRID = (0., .25, .5, .75, 1.)


def source_rank(arrays, grouped, district, members):
    """Mean new-site rank across *all* district candidates by lambda."""
    base, _, residual, novel, _ = arrays
    ids_by_district = {int(d): np.flatnonzero(district == d) for d in members}
    local_index = {d: {int(cell): j for j, cell in enumerate(ids)}
                   for d, ids in ids_by_district.items()}
    total = np.zeros(len(LAMBDA_GRID), np.float64)
    count = 0
    for (day, d), events in grouped.items():
        target = [candidate for event, candidate in events if event["new_site"]]
        if not target:
            continue
        ids = ids_by_district[d]
        js = [local_index[d][candidate] for candidate in target]
        b = base[day, ids]
        delta = residual[day, ids]*novel[day, ids]
        for i, strength in enumerate(LAMBDA_GRID):
            total[i] += float(pct(b+strength*delta)[js].sum())
        count += len(js)
    if not count:
        raise ValueError("No source new-site validation events")
    return {str(x): round(float(total[i]/count), 6)
            for i, x in enumerate(LAMBDA_GRID)}, count


def render(curves):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180, sharex=True)
    for ax, key, title in zip(axes, ("all_sites", "first_in_year_sites"),
                              ("All reported flood-site events", "Emerging first-in-year sites")):
        rows = curves[key]
        x = [0]+[100*r["fraction_all_cells"] for r in rows]
        for arm, color, name in (("history", "#8799a4", "Recent + nearby report history"),
                                 ("cnn", "#216d78", "History + site CNN residual")):
            y = [0]+[r[arm+"_capture_pct"] for r in rows]
            ax.plot(x, y, color=color, lw=2.2, marker="o", markersize=3.4,
                    label=name)
        ax.set_title(title, fontsize=11.5, weight="bold")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 90 if key == "all_sites" else 50)
        ax.set_xticks((0, 2, 5, 10, 20, 30, 50))
        ax.set_yticks((0, 20, 40, 60, 80) if key == "all_sites"
                      else (0, 10, 20, 30, 40, 50))
        ax.grid(alpha=.22)
        ax.set_xlabel("Top share of all district candidate cells reviewed")
        ax.set_ylabel("Reported site events captured (%)")
        ten = next(r for r in rows if r["fraction_all_cells"] == .10)
        ax.text(.04, .95,
                f"At 10% of all cells: {ten['history_capture_pct']:.2f}% history  |  "
                f"{ten['cnn_capture_pct']:.2f}% CNN",
                transform=ax.transAxes, fontsize=8.2, color="#173044",
                va="top", bbox={"facecolor": "white", "edgecolor": "#d9e3e6",
                                "boxstyle": "round,pad=0.28"})
    axes[0].legend(loc="lower right", fontsize=8.3, frameon=False)
    fig.suptitle("Bangkok Q4 2025 | no reserved discovery slots | 250 m cells",
                 fontsize=12, y=.99)
    fig.text(.5, .005,
             "Equal cell counts per district and day; reported sites are proxies, not verified flood extent.",
             ha="center", fontsize=8.3, color="#4c5c69")
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(PLOT, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    data = load_grid()
    sites = build(data["candidate"], data["district_id"])[250]
    flat = data["candidate_flat"]
    lookup = np.full(np.prod(SHAPE), -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    district = data["district_id"].ravel()[flat]
    selected = {r["fold"]: r for r in json.loads(RANK.read_text())["fold_selections"]}
    states = torch.load(MODEL, map_location="cpu", weights_only=True)["states"]
    folds = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    all_rows, choices = [], []
    for fold, members in enumerate(folds, 1):
        arm = selected[fold]["arm"]
        model = SpatialNovelCNN().eval()
        model.load_state_dict(states[fold][arm])
        static, _ = static_image(data, ~np.isin(district, members))
        source = np.setdiff1d(np.arange(len(data["districts"])), members)
        source_allowed = np.zeros(len(data["districts"]), bool)
        source_allowed[source] = True
        val_group = group_events(sites, source_allowed, VALID_START, TEST_START, lookup)
        val_arrays, _ = score_days(data, model, static, range(VALID_START, TEST_START))
        validation, n_val = source_rank(val_arrays, val_group, district, source)
        strength = max(LAMBDA_GRID, key=lambda x: (validation[str(x)], -x))
        choices.append({"fold": fold, "arm": arm, "source_new_site_events": n_val,
                        "source_mean_rank_by_strength": validation,
                        "selected_strength": strength})
        del val_arrays
        print("fold", fold, "source strength", strength,
              "new events", n_val, flush=True)
        q4_arrays, _ = score_days(data, model, static, range(TEST_START, N_DAYS))
        target_allowed = np.zeros(len(data["districts"]), bool)
        target_allowed[members] = True
        grouped = group_events(sites, target_allowed, TEST_START, N_DAYS, lookup)
        ids_by_district = {int(d): np.flatnonzero(district == d) for d in members}
        local_index = {d: {int(cell): j for j, cell in enumerate(ids)}
                       for d, ids in ids_by_district.items()}
        for (day, d), events in grouped.items():
            ids = ids_by_district[d]
            n = len(ids)
            base = q4_arrays[0][day, ids]
            learned = base + strength*q4_arrays[2][day, ids]*q4_arrays[3][day, ids]
            weights = {}
            for f in FRACTIONS:
                weights[f"history_{f:g}"] = top_weights(base, f*n)
                weights[f"cnn_{f:g}"] = top_weights(learned, f*n)
            for event, candidate in events:
                j = local_index[d][candidate]
                row = {"fold": fold, "day": event["day"], "district_index": d,
                       "row": event["row"], "col": event["col"],
                       "new_site_250m": int(event["new_site"]),
                       "own_cell_history_zero": int(q4_arrays[3][day, candidate]),
                       "nearest_prior_report_m": event["nearest_prior_report_m"]}
                row.update({name: float(weight[j]) for name, weight in weights.items()})
                all_rows.append(row)
        print("fold", fold, "Q4 events", sum(r["fold"] == fold for r in all_rows),
              flush=True)
        del q4_arrays
    if len(all_rows) != 2326 or sum(r["new_site_250m"] for r in all_rows) != 471:
        raise ValueError("Raw-site event targets changed")
    with EVENTS.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    curves = {}
    for key, subset in (("all_sites", all_rows),
                        ("first_in_year_sites", [r for r in all_rows if r["new_site_250m"]])):
        n = len(subset)
        curve = []
        for f in FRACTIONS:
            h = sum(r[f"history_{f:g}"] for r in subset)
            c = sum(r[f"cnn_{f:g}"] for r in subset)
            curve.append({"fraction_all_cells": f, "events": n,
                          "history_hits": round(h, 4), "cnn_hits": round(c, 4),
                          "history_capture_pct": round(100*h/n, 3),
                          "cnn_capture_pct": round(100*c/n, 3),
                          "cnn_minus_history_points": round(100*(c-h)/n, 3)})
        curves[key] = curve
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "task": "Full district all-cell queues without an unfamiliar-cell slot reserve",
              "history_score": "log1p(365d reports)+0.4 log1p(90d reports)+0.4 log1p(250m-smoothed 365d reports)",
              "model_score": "history_score + source-selected strength * CNN residual for cells with zero own 365d reports; familiar cells keep history_score",
              "source_selection": "Residual strength in [0,1] maximizes mean first-in-year site rank percentile across all cells on Jan-Sep 2025 source districts; CNN arm came from quota-free source selection. The range shrinks or preserves the trained residual without extrapolating it.",
              "fold_choices": choices, "curves": curves,
              "limits": ["Scores on unfamiliar and familiar cells are combined by a source-selected residual strength, not a forced slot quota",
                         "Q4 2025 and the 250m site definition were previously inspected",
                         "Reported site events are proxies for flood problems, not verified water extent",
                         "Cell-count budgets do not imply feasible field visits"]}
    RESULT.write_text(json.dumps(result, indent=2)+"\n")
    render(curves)
    print(json.dumps(curves, indent=2), flush=True)


if __name__ == "__main__":
    main()
