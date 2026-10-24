"""Select a frozen CNN residual multiplier on source-district validation queues."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_bangkok_spatial_cnn_citywide import HOLDOUT
from flood_citywide_history_ranker import N_DAYS, TEST_START, VALID_START, capture, load_grid
from flood_citywide_history_cnn_transfer import (
    OUT_MODEL, SpatialResidualCNN, day_inputs,
)
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval

ALPHAS = (0., .05, .1, .25, .5, 1.)
OUT_JSON = OUT / "q4_2025_history_cnn_transfer_calibrated.json"


def collect(data: dict, model: SpatialResidualCNN, days: range):
    base_rows, residual_rows, truth_rows = [], [], []
    for day in days:
        image, base = day_inputs(data, day)
        original = base[data["candidate"]].copy()
        predicted = model.full_scores(torch.from_numpy(image),
                                      base, data["candidate"])
        base_rows.append(original)
        residual_rows.append(predicted-original)
        truth_rows.append(data["reports"][day].ravel()[data["candidate_flat"]].copy())
    return np.stack(base_rows), np.stack(residual_rows), np.stack(truth_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_grid()
    state = torch.load(OUT_MODEL, weights_only=True)
    model = SpatialResidualCNN()
    model.load_state_dict(state["state"])
    model.eval()
    district_id = data["district_id"].ravel()[data["candidate_flat"]]
    heldout_ids = {i for i, district in enumerate(data["districts"])
                   if district["code"] in HOLDOUT}
    target = np.isin(district_id, list(heldout_ids))
    source = ~target

    validation_base, validation_residual, validation_truth = collect(
        data, model, range(VALID_START, TEST_START))
    validation = {}
    for alpha in ALPHAS:
        rows = capture(validation_base+alpha*validation_residual,
                       validation_truth, source)
        validation[f"{alpha:g}"] = rows["top_10pct_capture"]
    chosen = max(ALPHAS, key=lambda alpha: validation[f"{alpha:g}"])
    print("source validation", validation, "selected alpha", chosen, flush=True)
    del validation_base, validation_residual, validation_truth

    base, residual, truth = collect(data, model, range(TEST_START, N_DAYS))
    masks = {"eight_unseen_districts": target, "other_42_districts": source,
             "all_bangkok": np.ones(len(source), bool)}
    q4 = {}
    for scope, mask in masks.items():
        rows = {"fixed_recent_nearby": capture(base, truth, mask),
                "cnn_raw_alpha_1": capture(base+residual, truth, mask),
                "cnn_selected": capture(base+chosen*residual, truth, mask)}
        rows["selected_minus_fixed_95pct_weekly_block_interval"] = paired_interval(
            rows["cnn_selected"], rows["fixed_recent_nearby"])
        q4[scope] = rows
    if q4["all_bangkok"]["fixed_recent_nearby"]["top_10pct_capture"] != .48168:
        raise ValueError("Frozen citywide baseline differs")
    output = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "daily top-10% Traffy report-positive cell-days",
        "model": "frozen 15x15 dilated CNN residual on fixed recent+nearby score",
        "selection_scope": "42 source districts, Jan-Sep 2025; target districts excluded",
        "candidate_multipliers": ALPHAS,
        "source_validation_capture": validation,
        "selected_multiplier": chosen,
        "q4": q4,
        "limitations": ["Q4 2025 inspected in prior project work",
                        "Reports are positive problem sites, not verified flood extent",
                        "Unreported cells are unknown, not verified dry"],
    }
    OUT_JSON.write_text(json.dumps(output, indent=2)+"\n")
    for scope in masks:
        print(scope, {name: q4[scope][name]["top_10pct_capture"]
                      for name in ("fixed_recent_nearby", "cnn_raw_alpha_1",
                                   "cnn_selected")}, flush=True)
    print("saved", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
