"""Test ESA WorldCover classes in the six-region GISTDA spatial CNN.

Uses the original geographic folds and dates, then compares built-only,
separate land-cover classes, and land-cover plus pre-event Sentinel-2 NDVI.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_gistda_multiregion_data import CACHE, CENTERS, OUTPUT, SIZE
from flood_gistda_multiregion_train import (
    TEST_DATE, aggregate, evaluate, load_data, normalize, scenes, train_one,
)
from flood_gistda_multiregion_worldcover import CHANNELS
from flood_gistda_multiregion_smap import CHANNEL as SMAP_CHANNEL


VARIANTS = ("built", "cover", "cover_s2", "cover_smap")


def load_variant(variant: str) -> tuple[dict, list[str], tuple[str, ...]]:
    regions, dates, names = load_data(with_sentinel=variant == "cover_s2")
    for name in CENTERS:
        with np.load(CACHE / f"{name}_worldcover_2021_250m.npz") as saved:
            cover = saved["values"].astype(np.float32)
            if tuple(saved["names"].astype(str)) != CHANNELS:
                raise ValueError(f"WorldCover channel mismatch: {name}")
        if cover.shape != (len(CHANNELS), SIZE, SIZE):
            raise ValueError(f"WorldCover shape mismatch: {name}")
        if variant == "built":
            cover = cover[:1]
        regions[name]["x"] = np.concatenate((regions[name]["x"],
            np.broadcast_to(cover, (len(dates), *cover.shape))), axis=1).copy()
    names = names + (CHANNELS[:1] if variant == "built" else CHANNELS)
    if variant == "cover_smap":
        for name in CENTERS:
            with np.load(CACHE / f"{name}_smap_2026_250m.npz") as saved:
                smap = saved["values"].astype(np.float32)
                if tuple(saved["names"].astype(str)) != (SMAP_CHANNEL,):
                    raise ValueError(f"SMAP channel mismatch: {name}")
                if list(saved["dates"].astype(str)) != dates:
                    raise ValueError(f"SMAP dates mismatch: {name}")
            if smap.shape != (len(dates), 1, SIZE, SIZE):
                raise ValueError(f"SMAP shape mismatch: {name}")
            regions[name]["x"] = np.concatenate((regions[name]["x"], smap), axis=1)
        names += (SMAP_CHANNEL,)
    return regions, dates, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS + ("all",), default="all")
    args = parser.parse_args()
    torch.set_num_threads(4)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        fold_training = {}
        normalization = {}
        evaluation = []
        for holdout in CENTERS:
            train_regions = tuple(name for name in CENTERS if name != holdout)
            regions, dates, names = load_variant(variant)
            normalization[holdout] = normalize(regions, train_regions)
            run_name = f"cover_{variant}_holdout_{holdout}"
            model, fold_training[holdout] = train_one(
                "cnn", regions, dates, train_regions, names, run_name)
            for split, pairs in (
                ("geographic_holdout", scenes(regions, (holdout,), tuple(range(len(dates))))),
                ("later_date_test", scenes(regions, train_regions, (TEST_DATE,))),
            ):
                rows = evaluate(model, regions, pairs, dates, f"cnn_{variant}")
                for row in rows:
                    row["split"] = split
                    row["holdout_region"] = holdout
                evaluation.extend(rows)
            print("completed", variant, holdout, flush=True)
        summary = {
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "variant": variant, "feature_names": names,
            "source": "ESA WorldCover 2021 via Planetary Computer STAC" +
                      ("; GISTDA weekly SMAP" if variant == "cover_smap" else ""),
            "target": "conditional GISTDA positive-area density, not wet/dry probability",
            "folds": list(CENTERS), "training": fold_training,
            "normalization_from_training": normalization,
            "summary": aggregate(evaluation),
            "geographic_fold_summary": [
                {**row, "holdout_region": holdout}
                for holdout in CENTERS
                for row in aggregate([r for r in evaluation
                    if r["split"] == "geographic_holdout"
                    and r["holdout_region"] == holdout])
            ],
            "limitations": [
                "WorldCover 2021 built class is not measured impervious area or soil infiltration",
                "Patch centers were selected after inspecting positive GISTDA coverage",
                "GISTDA cells outside mapped polygons are unknown, not verified dry",
                "The 2026 product dates share repeated water geography and varying satellite coverage",
                *(["October 20 uses a 15-day-old October 5 SMAP product"]
                  if variant == "cover_smap" else []),
            ],
        }
        (OUTPUT / f"gistda_cover_{variant}_metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        with (OUTPUT / f"gistda_cover_{variant}_evaluation.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(evaluation[0]))
            writer.writeheader()
            writer.writerows(evaluation)
        print("RESULT", variant, summary["summary"], flush=True)


if __name__ == "__main__":
    main()
