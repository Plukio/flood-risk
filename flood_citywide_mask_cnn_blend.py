"""Source-selected rank blend of the saved masked CNN and nearby history.

The existing CNN is frozen. Each component is converted to a within-day,
within-fold percentile among first-in-a-year candidates, so the blend does not
depend on the arbitrary CNN logit scale. The same fixed 20% reserve is used.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.stats import rankdata

from flood_citywide_history_ranker import N_DAYS, TEST_START, VALID_START, load_grid
from flood_citywide_spatial_mask_augmentation import OUT_JSON as OLD_JSON, OUT_MODEL as OLD_MODEL
from flood_citywide_spatial_novel_queue import (
    SpatialNovelCNN, collect, queue_weights, static_image,
)
from flood_multi_target_citywide_data import OUT

OUT_JSON = OUT / "q4_2025_mask_cnn_neighbor_blend_fivefold.json"
ALPHAS = (0., .25, .5, .75, 1.)  # CNN rank share; zero is nearby history.
QUOTA = .20


def percentile(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=np.float32)
    return ((rankdata(values, method="average") - 1) / (len(values) - 1)).astype(np.float32)


def score_days(arrays, scope: np.ndarray, alphas=ALPHAS):
    base, simple, cnn, novel, truth = arrays
    indices = np.flatnonzero(scope)
    summary = {str(a): {"positive": 0, "novel_positive": 0,
                         "captured": 0., "novel_captured": 0.,
                         "daily_novel_positive": [], "daily_novel_captured": []}
               for a in alphas}
    for d in range(len(truth)):
        n = novel[d, indices]
        y = truth[d, indices]
        s = np.zeros(len(indices), np.float32)
        c = np.zeros(len(indices), np.float32)
        s[n] = percentile(simple[d, indices][n])
        c[n] = percentile(cnn[d, indices][n])
        for a in alphas:
            score = (1-a)*s+a*c
            w = queue_weights(base[d, indices], score, n, QUOTA)
            row = summary[str(a)]
            np_count = int(y[n].sum())
            nh = float(np.dot(w, n*y))
            row["positive"] += int(y.sum())
            row["novel_positive"] += np_count
            row["captured"] += float(np.dot(w, y))
            row["novel_captured"] += nh
            row["daily_novel_positive"].append(np_count)
            row["daily_novel_captured"].append(round(nh, 5))
    for row in summary.values():
        row["overall_capture"] = round(row["captured"]/row["positive"], 5)
        row["novel_capture"] = round(row["novel_captured"]/row["novel_positive"], 5)
        row["captured"] = round(row["captured"], 5)
        row["novel_captured"] = round(row["novel_captured"], 5)
    return summary


def main():
    torch.set_num_threads(4)
    data = load_grid()
    old = json.loads(OLD_JSON.read_text())
    states = torch.load(OLD_MODEL, weights_only=True)["states"]
    district = data["district_id"].ravel()[data["candidate_flat"]]
    district_folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    folds = []
    for fold_number, ids in enumerate(district_folds, 1):
        target = np.isin(district, ids)
        source = ~target
        static, _ = static_image(data, source)
        model = SpatialNovelCNN()
        model.load_state_dict(states[fold_number]["synthetic_center_block"])
        model.eval()
        print("fold", fold_number, "scoring source", flush=True)
        validation = score_days(collect(data, model, static,
                               range(VALID_START, TEST_START)), source)
        selected = max(ALPHAS, key=lambda a: (validation[str(a)]["novel_capture"], -a))
        print("fold", fold_number, "selected", selected, "scoring Q4", flush=True)
        q4 = score_days(collect(data, model, static,
                        range(TEST_START, N_DAYS)), target)
        folds.append({"fold": fold_number, "district_codes":
                      [data["districts"][j]["code"] for j in ids],
                      "source_validation": validation, "source_selected_alpha": selected,
                      "q4": q4})
        print("fold", fold_number, "Q4", {a:q4[str(a)]["novel_capture"]
                                              for a in ALPHAS}, flush=True)

    def pool(choice):
        rows = [f["q4"][str(choice(f))] for f in folds]
        hits = sum(r["captured"] for r in rows)
        novel_hits = sum(r["novel_captured"] for r in rows)
        positives = sum(r["positive"] for r in rows)
        novel_positives = sum(r["novel_positive"] for r in rows)
        return {"positive": positives, "novel_positive": novel_positives,
                "captured": round(hits, 5), "novel_captured": round(novel_hits, 5),
                "overall_capture": round(hits/positives, 5),
                "novel_capture": round(novel_hits/novel_positives, 5),
                "daily_novel_positive": np.sum([r["daily_novel_positive"] for r in rows],
                                               axis=0).astype(int).tolist(),
                "daily_novel_captured": np.sum([r["daily_novel_captured"] for r in rows],
                                               axis=0).round(5).tolist()}

    pooled = {str(a): pool(lambda f, a=a: a) for a in ALPHAS}
    pooled["source_selected"] = pool(lambda f: f["source_selected_alpha"])
    expected = old["pooled_fold_queues"]
    if (pooled["0.0"]["positive"] != 7209 or
        pooled["0.0"]["novel_positive"] != 1664 or
        abs(pooled["0.0"]["novel_capture"]-
            expected["simple_neighbor"]["novel_capture"]) > 1e-5 or
        abs(pooled["1.0"]["novel_capture"]-
            expected["synthetic_center_block"]["novel_capture"]) > 1e-5):
        raise ValueError("Saved queues did not reproduce")
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "task": "Q4 2025 first-in-a-year report-positive cell capture",
              "queue": "five held-out district folds, 10% daily capacity, 20% reserved for no own-cell prior-365d reports",
              "method": "within-day rank percentile blend among novel candidates; alpha is CNN share",
              "selection": "highest Jan-Sep 2025 source-district first-in-a-year capture; ties favor simple history",
              "alphas": ALPHAS, "folds": folds, "pooled_q4": pooled,
              "caution": "Q4 has been repeatedly examined; this is exploratory, not an untouched prospective test."}
    OUT_JSON.write_text(json.dumps(result, indent=2)+"\n")
    print("pooled", {k:(v["novel_capture"],v["overall_capture"])
                     for k,v in pooled.items()}, flush=True)
    print("saved", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
