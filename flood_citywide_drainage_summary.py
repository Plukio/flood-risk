"""Summarize three repeated fits on the same held-out Q4 days."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from flood_citywide_drainage_adapter import OUT, OUT_JSON

OUTPUT = OUT / "q4_2025_bma_drainage_adapter_three_seed_summary.json"
SEEDS = (0, 10000, 20000)


def main():
    rows = []
    for seed in SEEDS:
        suffix = f"_seed_{seed}" if seed else ""
        report = json.loads(OUT_JSON.with_name(OUT_JSON.stem+suffix+".json").read_text())
        pooled = report["pooled_heldout_q4"]
        def short(row):
            return {"isolated_cell_hits": round(row["isolated_hits"], 2),
                    "isolated_group_hits": round(row["group_hits"], 2),
                    "isolated_cell_capture": row["isolated_cell_capture"],
                    "isolated_group_capture": row["isolated_group_capture"]}
        rows.append({"seed_offset": seed,
                     "frozen_bma": short(pooled["real"]["0.0"]),
                     "real_drainage": short(pooled["real"]["source_selected"]),
                     "shuffled_outlets": short(pooled["shuffled"]["source_selected"]),
                     "real_alphas_by_fold": [f["arms"]["real"]["selected_alpha"]
                                            for f in report["folds"]],
                     "shuffled_alphas_by_fold": [f["arms"]["shuffled"]["selected_alpha"]
                                                for f in report["folds"]]})
    averages = {}
    for key in ("frozen_bma", "real_drainage", "shuffled_outlets"):
        averages[key] = {metric: round(float(np.mean([r[key][metric] for r in rows])), 3)
                         for metric in rows[0][key]}
    result = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "repeats_share_the_same_q4_dates": True,
              "denominators": {"isolated_cells": 332, "touching_groups": 158},
              "seeds": rows, "mean_over_fits": averages,
              "interpretation": "Repeated fits are sensitivity checks, not independent test sets."}
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
