"""Audit whether historical report sources exist on the pretraining paths."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.sparse import csr_matrix

from flood_citywide_history_ranker import N_DAYS, TEST_START, load_grid
from flood_citywide_isolated_site_ranker import isolated
from flood_citywide_route_pretrain import OUT, terrain

OUT_JSON = OUT / "q4_2025_isolated_path_history_coverage.json"


def main():
    data = load_grid()
    _, node_flat, down, _ = terrain(data)
    rows, cols = [], []
    for source in range(len(node_flat)):
        target = source
        for hop in range(1, 7):
            target = int(down[target])
            if target < 0:
                break
            if hop >= 2:
                rows.append(target)
                cols.append(source)
    path = csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                      shape=(len(node_flat), len(node_flat)))
    counts = {"isolated_positive_cell_days": 0,
              "isolated_positive_with_prior_report_on_path": 0,
              "isolated_positive_with_prior_report_within_1500m_grid_radius": 0,
              "isolated_eligible_cell_days": 0,
              "isolated_eligible_with_prior_report_on_path": 0,
              "isolated_eligible_with_prior_report_within_1500m_grid_radius": 0}
    for day in range(TEST_START, N_DAYS):
        prior = (data["prefix"][day] - data["prefix"][day-365]) > 0
        path_history = (path @ prior.ravel()[node_flat].astype(np.uint8)) > 0
        nearby_history = maximum_filter(prior.astype(np.uint8),
                                        size=13, mode="constant").ravel()[node_flat] > 0
        eligible = isolated(data, day).ravel()[node_flat]
        positive = (data["reports"][day].ravel()[node_flat] > 0) & eligible
        counts["isolated_positive_cell_days"] += int(positive.sum())
        counts["isolated_positive_with_prior_report_on_path"] += int(
            (positive & path_history).sum())
        counts["isolated_positive_with_prior_report_within_1500m_grid_radius"] += int(
            (positive & nearby_history).sum())
        counts["isolated_eligible_cell_days"] += int(eligible.sum())
        counts["isolated_eligible_with_prior_report_on_path"] += int(
            (eligible & path_history).sum())
        counts["isolated_eligible_with_prior_report_within_1500m_grid_radius"] += int(
            (eligible & nearby_history).sum())
    if counts["isolated_positive_cell_days"] != 332:
        raise ValueError("Frozen isolated Q4 denominator changed")
    result = {"built_utc": datetime.now(timezone.utc).isoformat(
                  timespec="seconds"),
              "question": "How often does a strict isolated site have a prior report on a 2-6-hop D8 upstream path?",
              "grid_m": 250,
              "history_window_days": 365,
              "path_hops": [2, 6],
              "nearby_definition": "Any prior report within six grid cells Chebyshev radius (<=1500 m along axes), with isolated 3x3 history already empty",
              "path_pairs": len(rows),
              "path_receiver_cells": int(np.unique(rows).size),
              "counts": counts,
              "limitations": ["D8 is a terrain proxy, not proven urban drainage",
                              "Prior report history records complaints, not observed wetness",
                              "The nearby square is an exposure diagnostic, not the exact Gaussian history score"]}
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2)+"\n")
    print(result)


if __name__ == "__main__":
    main()
