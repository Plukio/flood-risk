"""Summarize matched BMA road-bag transfer runs without selecting on Q4."""
from __future__ import annotations

import json

import numpy as np

from flood_citywide_bma_road_pretrain import OUT_JSON

SEEDS = (0, 10000, 20000)
OUT = OUT_JSON.with_name("q4_2025_bma_road_pretrain_mask_three_seed_summary.json")


def selected(arm):
    return arm["heldout_q4"][str(arm["selected_alpha"])]


def summarize():
    runs = []
    for seed in SEEDS:
        path = OUT_JSON.with_name(OUT_JSON.stem +
            ("" if seed == 0 else f"_seed_{seed}") + ".json")
        runs.append(json.loads(path.read_text()))
    cells = {name: [] for name in ("real", "shuffled", "path", "history")}
    positive = []
    per_seed = []
    for seed, run in zip(SEEDS, runs):
        fold_hits = {name: [] for name in cells}
        fold_positive = []
        rows = {name: [] for name in cells}
        for fold in run["folds"]:
            arm = fold["arms"]
            chosen = {
                "real": selected(arm["bma_real"]),
                "shuffled": selected(arm["bma_shuffled"]),
                "path": selected(fold["saved_path_control"]),
                "history": arm["bma_real"]["heldout_q4"]["0.0"],
            }
            counts = np.asarray(chosen["real"]["daily_isolated_positive"])
            if any(not np.array_equal(counts,
                    row["daily_isolated_positive"]) for row in chosen.values()):
                raise ValueError("Matched Q4 positive labels differ")
            fold_positive.append(counts)
            for name, row in chosen.items():
                rows[name].append(row)
                fold_hits[name].append(np.asarray(row["daily_isolated_hits"]))
        pos = np.sum(fold_positive, axis=0)
        positive.append(pos)
        for name in cells:
            cells[name].append(np.sum(fold_hits[name], axis=0))
        summary = {name: {"cells": round(float(sum(r["isolated_hits"]
                                                   for r in rows[name])), 2),
                          "groups": round(float(sum(r["group_hits"]
                                                    for r in rows[name])), 2)}
                   for name in cells}
        per_seed.append({"seed_offset": seed, **summary})
    if not all(np.array_equal(positive[0], p) for p in positive[1:]):
        raise ValueError("Q4 positives changed across seeds")
    if int(positive[0].sum()) != 332:
        raise ValueError("Frozen isolated Q4 cohort changed")
    blocks = [np.arange(i, min(i+7, len(positive[0])))
              for i in range(0, len(positive[0]), 7)]
    rng = np.random.default_rng(714)
    intervals = {}
    real = np.mean(cells["real"], axis=0)
    for name in ("path", "shuffled", "history"):
        other = np.mean(cells[name], axis=0)
        point = float((real.sum()-other.sum())/positive[0].sum())
        draws = []
        for _ in range(10000):
            idx = np.concatenate([blocks[k] for k in rng.integers(
                len(blocks), size=len(blocks))])
            denom = positive[0][idx].sum()
            if denom:
                draws.append(float((real[idx]-other[idx]).sum()/denom))
        intervals[f"real_minus_{name}"] = {
            "point_percentage_points": round(100*point, 3),
            "q4_seven_day_block_95pct_interval_percentage_points":
                [round(float(100*x), 3) for x in np.quantile(draws, [.025, .975])],
        }
    output = {
        "seed_offsets": SEEDS,
        "per_seed": per_seed,
        "mean_hits_over_same_q4": {
            name: {"cells": round(float(np.mean([r[name]["cells"] for r in per_seed])), 2),
                   "groups": round(float(np.mean([r[name]["groups"] for r in per_seed])), 2)}
            for name in cells},
        "positive_cell_days": int(positive[0].sum()),
        "intervals": intervals,
        "interpretation": "The three seeds reuse the same held-out Q4 dates; date-block intervals are exploratory and do not create three independent test sets",
    }
    OUT.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    summarize()
