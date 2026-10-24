"""Ablate adding spaced 2025 positive maps to four-city GISTDA training."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_gistda_multiregion_data import CACHE, OUTPUT, CENTERS, SIZE
from flood_gistda_multiregion_2025_test import PAIRS, features, rain
from flood_gistda_multiregion_data import specs
from flood_gistda_multiregion_train import (TRAIN_DATES, VAL_DATES, TEST_DATE,
                                             load_data, scenes, train_one,
                                             evaluate, aggregate)


OLD_DATES = tuple(sorted({day for _, day in PAIRS}))
PAIR_SET = set(PAIRS)


def load_augmented(with_sentinel: bool) -> tuple[dict, list[str], tuple[str, ...]]:
    regions, dates, names = load_data(with_sentinel)
    all_specs = specs()
    for name in CENTERS:
        weather = rain(name, all_specs[name]) if any(n == name for n, _ in PAIRS) else None
        old_x, old_wet = [], []
        for day in OLD_DATES:
            if (name, day) in PAIR_SET:
                with np.load(CACHE / f"{name}_wet_{day}_250m.npz") as saved:
                    wet = saved["wet_area_m2"].astype(np.float32)
                if wet.sum() <= 0:
                    raise ValueError(f"Missing mapped water: {name} {day}")
                x = features(name, day, weather, with_sentinel)[0]
            else:
                wet = np.zeros((SIZE, SIZE), np.float32)
                x = np.zeros((len(names), SIZE, SIZE), np.float32)
            old_x.append(x)
            old_wet.append(wet)
        regions[name]["x"] = np.concatenate((regions[name]["x"], np.stack(old_x)))
        regions[name]["wet"] = np.concatenate((regions[name]["wet"], np.stack(old_wet)))
    return regions, dates + [str(d) for d in OLD_DATES], names


def normalize_train_only(regions: dict, train_regions: tuple[str, ...]) -> dict:
    # The held-out city's 2025 and 2026 data never affect standardization.
    samples = []
    for name in train_regions:
        samples.extend(regions[name]["x"][list(TRAIN_DATES)])
        samples.extend(regions[name]["x"][8+i] for i in range(len(OLD_DATES))
                       if regions[name]["wet"][8+i].sum() > 0)
    values = np.stack(samples)
    mean = values.mean(axis=(0, 2, 3))
    std = np.maximum(values.std(axis=(0, 2, 3)), .02)
    for item in regions.values():
        item["x"] = np.clip((item["x"] - mean[None, :, None, None]) /
                            std[None, :, None, None], -5, 5).astype(np.float32)
    return {"mean": mean.tolist(), "std": std.tolist()}


def main() -> None:
    torch.set_num_threads(4)
    old = json.loads((OUTPUT / "gistda_multiregion_training_metrics.json").read_text())
    training = {}
    normalization = {}
    rows = []
    train_dates = TRAIN_DATES + tuple(range(8, 8 + len(OLD_DATES)))
    for holdout in CENTERS:
        train_regions = tuple(name for name in CENTERS if name != holdout)
        for with_sentinel in (False, True):
            regions, dates, names = load_augmented(with_sentinel)
            suffix = "s2" if with_sentinel else "base"
            run_name = f"aug2025_holdout_{holdout}_{suffix}"
            normalization[run_name] = normalize_train_only(regions, train_regions)
            model, training[run_name] = train_one(
                "cnn", regions, dates, train_regions, names, run_name, train_dates)
            for split, pairs in (
                ("geographic_holdout", scenes(regions, (holdout,), tuple(range(8)))),
                ("geographic_holdout_2025", scenes(regions, (holdout,),
                                                   tuple(range(8, 8 + len(OLD_DATES))))),
                ("later_date_test", scenes(regions, train_regions, (TEST_DATE,)))):
                output = evaluate(model, regions, pairs, dates,
                                  "cnn_s2_2025aug" if with_sentinel else "cnn_base_2025aug")
                for row in output:
                    row["split"] = split
                    row["holdout_region"] = holdout
                rows.extend(output)
    fold = []
    for holdout in CENTERS:
        for row in aggregate([x for x in rows if x["split"] == "geographic_holdout"
                              and x["holdout_region"] == holdout]):
            row["holdout_region"] = holdout
            fold.append(row)
    summary = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "added_positive_maps": [(n, str(d)) for n, d in PAIRS],
               "training_dates_2026": [dates[i] for i in TRAIN_DATES],
               "training_dates_2025": [str(d) for d in OLD_DATES],
               "selection_dates": [dates[i] for i in VAL_DATES],
               "test": "2026 positive mapped-water area in a completely held-out region",
               "note": "2025 patch dates were selected after overlap inspection; no-polygon cells remain unknown and are skipped as training scenes. Each region is held out for both years.",
               "normalization_from_training": normalization,
               "training": training, "summary": aggregate(rows),
               "geographic_fold_summary": fold,
               "original_2026_training_summary": [x for x in old["summary"]
                                                  if x["split"] == "geographic_holdout"]}
    (OUTPUT / "gistda_multiregion_2025_augmented_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    for row in summary["summary"]:
        print(row, flush=True)


if __name__ == "__main__":
    main()
