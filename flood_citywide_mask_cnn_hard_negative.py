"""Train the existing masked CNN with harder comparison cells.

Only the pool for history-based hard negatives changes: from the highest 20%
to the highest 5% of nearby-history candidate scores. Model selection stays on
source-district 2025 dates; Q4 target districts are scored once afterward.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS, TEST_START, VALID_START, load_grid
from flood_citywide_history_cnn_transfer import SEED
from flood_citywide_mask_cnn_blend import OUT_JSON as CONTROL_JSON, score_days
from flood_citywide_spatial_mask_augmentation import fit
from flood_citywide_spatial_novel_queue import collect, static_image
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_mask_cnn_hard_negative_fivefold.json"
OUT_MODEL = OUT / "q4_2025_mask_cnn_hard_negative_fivefold.pt"


def main():
    torch.set_num_threads(4)
    data = load_grid()
    control = json.loads(CONTROL_JSON.read_text())
    district = data["district_id"].ravel()[data["candidate_flat"]]
    district_folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    folds, states = [], {}
    for fold_number, ids in enumerate(district_folds, 1):
        target = np.isin(district, ids)
        source = ~target
        static, _ = static_image(data, source)
        print("fold", fold_number, "training hard-top-5% CNN", flush=True)
        model, audit = fit(data, source, static, SEED+fold_number*1000,
                           block_mask=True, hard_fraction=.05, hard_floor=25)
        states[fold_number] = {k:v.cpu() for k,v in model.state_dict().items()}
        val = score_days(collect(data, model, static,
                         range(VALID_START, TEST_START)), source, alphas=(1.,))["1.0"]
        original = control["folds"][fold_number-1]["source_validation"]["1.0"]
        chosen = "hard_top5" if val["novel_capture"] > original["novel_capture"] else "original_top20"
        q4 = score_days(collect(data, model, static,
                        range(TEST_START, N_DAYS)), target, alphas=(1.,))["1.0"]
        folds.append({"fold": fold_number, "fit": audit,
                      "source_validation": {"hard_top5":val, "original_top20":original},
                      "source_selected_arm":chosen,
                      "q4": {"hard_top5":q4,
                             "original_top20":control["folds"][fold_number-1]["q4"]["1.0"]}})
        print("fold", fold_number, "selection", chosen,
              "Q4 novel", q4["novel_capture"], flush=True)

    def pool(choice):
        rows = [f["q4"][choice(f)] for f in folds]
        novel_hits = sum(r["novel_captured"] for r in rows)
        hits = sum(r["captured"] for r in rows)
        novel_positive = sum(r["novel_positive"] for r in rows)
        positive = sum(r["positive"] for r in rows)
        return {"positive":positive, "novel_positive":novel_positive,
                "captured":round(hits,5), "novel_captured":round(novel_hits,5),
                "overall_capture":round(hits/positive,5),
                "novel_capture":round(novel_hits/novel_positive,5)}

    pooled = {"hard_top5":pool(lambda f:"hard_top5"),
              "original_top20":pool(lambda f:"original_top20"),
              "source_selected":pool(lambda f:f["source_selected_arm"])}
    if (pooled["original_top20"]["novel_positive"] != 1664 or
        pooled["original_top20"]["novel_capture"] !=
            control["pooled_q4"]["1.0"]["novel_capture"]):
        raise ValueError("Original control changed")
    result = {"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "task":"Q4 2025 first-in-a-year report-positive cell capture",
              "change":"hard-history comparison pool top 5% instead of top 20%; same masked CNN, masks, epochs, sampling seed and fixed queue",
              "selection":"per-fold Jan-Sep 2025 source-district novel capture; ties retain original",
              "folds":folds, "pooled_q4":pooled,
              "caution":"Q4 has been repeatedly examined; exploratory result only"}
    OUT_JSON.write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"states":states},OUT_MODEL)
    print("pooled",pooled,flush=True)
    print("saved",OUT_JSON,OUT_MODEL,flush=True)


if __name__ == "__main__":
    main()
