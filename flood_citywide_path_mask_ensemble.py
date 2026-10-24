"""Source-selected ensemble of three saved path-pretrained masked CNNs.

Run: .venv/bin/python flood_citywide_path_mask_ensemble.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_citywide_history_ranker import N_DAYS, TEST_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import (
    ALPHAS, ContextResidualCNN, pool, score_days,
)
from flood_citywide_path_pretrain_mask_transfer import OUT_MODEL as BASE_MODEL
from flood_citywide_route_pretrain import OUT, lidar_static, terrain

SEED_OFFSETS = (0, 10000, 20000)
OUT_JSON = OUT / "q4_2025_path_mask_three_seed_ensemble_fivefold.json"


class MeanResidual:
    def __init__(self, models: list[ContextResidualCNN]):
        self.models = models

    def full_scores(self, image: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        return np.mean([m.full_scores(image, candidate) for m in self.models],
                       axis=0).astype(np.float32)


def main():
    torch.set_num_threads(4)
    data = load_grid()
    elevation, _, _, _ = terrain(data)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    saved = []
    for offset in SEED_OFFSETS:
        suffix = "" if offset == 0 else f"_seed_{offset}"
        path = BASE_MODEL.with_name(BASE_MODEL.stem+suffix+".pt")
        saved.append(torch.load(path, map_location="cpu", weights_only=True))
    records = []
    for fold_id, ids in enumerate(folds, 1):
        source = ~np.isin(district, ids)
        static, norm = lidar_static(data, source, elevation)
        models = []
        for state in saved:
            model = ContextResidualCNN()
            model.load_state_dict(state["states"][fold_id]["path_pretrained"])
            model.eval()
            models.append(model)
        ensemble = MeanResidual(models)
        validation = score_days(data, source, static, ensemble,
                                range(VALID_START, TEST_START))
        alpha = max(ALPHAS, key=lambda a: (
            validation[str(a)]["isolated_group_capture"],
            validation[str(a)]["isolated_cell_capture"], -a))
        test = score_days(data, ~source, static, ensemble,
                          range(TEST_START, N_DAYS))
        records.append({"fold": fold_id,
                        "heldout_district_codes": [
                            data["districts"][j]["code"] for j in ids],
                        "normalization": norm,
                        "source_validation": validation,
                        "selected_alpha": alpha,
                        "heldout_q4": test})
        print("fold", fold_id, "alpha", alpha,
              "cells", test[str(alpha)]["isolated_hits"],
              "groups", test[str(alpha)]["group_hits"], flush=True)
    pooled = {str(a): pool([r["heldout_q4"][str(a)] for r in records])
              for a in ALPHAS}
    pooled["source_selected"] = pool([
        r["heldout_q4"][str(r["selected_alpha"])] for r in records])
    base = pooled["0.0"]
    if (base["isolated_positive"], base["isolated_groups"],
        base["isolated_hits"], base["group_hits"]) != (332,158,65.,44.):
        raise ValueError("Frozen strict isolated queue changed")
    result = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": "Does averaging three path-pretrained, target-masked CNN residuals beat distant report history on the strict isolated queue?",
        "seed_offsets": SEED_OFFSETS,
        "ensemble": "Arithmetic mean of three frozen residual score maps; no retraining",
        "selection": "Correction strength chosen per fold on January-September 2025 source-district isolated group capture, then cell capture, then smaller strength",
        "target": "Traffy report-positive 250m cells, not verified flooding",
        "queue": "10% of heldout district-fold cells per day, 20% reserved for strict isolated candidates",
        "folds": records, "pooled_heldout_q4": pooled,
        "limitations": ["Q4 2025 repeatedly inspected; exploratory",
                        "Unreported cells are unknown, not verified dry",
                        "Ensembling does not prove learned water flow or rain scenario skill"],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2)+"\n")
    print("pooled", {key:(row["isolated_hits"],row["group_hits"])
                     for key,row in pooled.items()}, flush=True)
    print("saved", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
