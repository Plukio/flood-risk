"""Test whether 2025 province maps improve fixed 2026 Bangkok transfer."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_gistda_multiregion_data import CACHE, CENTERS, OUTPUT, SIZE
from flood_gistda_multiregion_train import (
    TRAIN_DATES, evaluate, normalize, scenes, train_one,
)
from flood_gistda_province_expand_train import (
    FEATURE_NAMES, STAGES, TARGET, load_all,
)


AUDIT = OUTPUT / "province_2025_extra_data_audit.json"
DEST = OUTPUT / "province_expansion_2025_augmented_bangkok_metrics.json"
STAGE_NAMES = ("original_5", "plus_10", "plus_15")


def append_2025(regions: dict, dates: list[str]) -> list[str]:
    extra_dates = json.loads(AUDIT.read_text())["dates"]
    for name, item in regions.items():
        if name == TARGET:
            continue
        with np.load(CACHE / f"{name}_wet_2025_extra_250m.npz") as saved:
            wet = saved["wet_area_m2"].astype(np.float32)
            if list(saved["dates"].astype(str)) != extra_dates:
                raise ValueError(f"Extra label dates mismatch: {name}")
        with np.load(CACHE / f"{name}_rain_features_2025_extra.npz") as saved:
            rain = saved["values"].astype(np.float32)
            if list(saved["dates"].astype(str)) != extra_dates:
                raise ValueError(f"Extra rain dates mismatch: {name}")
        with np.load(CACHE / f"{name}_worldcover_2021_250m.npz") as saved:
            cover = saved["values"].astype(np.float32)
        static = item["static"]
        if wet.shape != (len(extra_dates), SIZE, SIZE) or rain.shape != (len(extra_dates), 7):
            raise ValueError(f"Extra data shape mismatch: {name}")
        months = np.asarray([int(day[5:7]) for day in extra_dates])
        global_values = np.concatenate((rain,
            np.sin(2 * np.pi * months[:, None] / 12),
            np.cos(2 * np.pi * months[:, None] / 12)), axis=1)
        x_extra = np.concatenate((
            np.broadcast_to(static, (len(extra_dates), 7, SIZE, SIZE)),
            np.broadcast_to(global_values[:, :, None, None],
                            (len(extra_dates), 9, SIZE, SIZE)),
            np.broadcast_to(cover, (len(extra_dates), 6, SIZE, SIZE))), axis=1)
        item["x"] = np.concatenate((item["x"], x_extra)).astype(np.float32)
        item["wet"] = np.concatenate((item["wet"], wet)).astype(np.float32)
    return dates + extra_dates


def main() -> None:
    torch.set_num_threads(4)
    old_external = tuple(name for name in CENTERS if name != TARGET)
    report = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "target": "2026 GISTDA mapped-water positive area in held-out Nong Chok",
              "feature_names": FEATURE_NAMES,
              "selection": "2025 extra dates chosen after earlier 2026 Bangkok runs; exploratory, not a pristine holdout",
              "year_split": "2025 province labels plus five early-2026 dates train; later-2026 dates from training provinces select epoch; all 2026 Nong Chok dates evaluate",
              "stages": {}}
    for base_name in STAGE_NAMES:
        stage = base_name + "_plus_2025"
        train_regions = old_external + STAGES[base_name]
        regions, dates = load_all()
        dates = append_2025(regions, dates)
        stats = normalize(regions, train_regions)
        train_dates = TRAIN_DATES + tuple(range(8, len(dates)))
        model, training = train_one("cnn", regions, dates, train_regions,
                                    FEATURE_NAMES, f"expanded_bkk_{stage}",
                                    train_dates=train_dates)
        rows = evaluate(model, regions, scenes(regions, (TARGET,), tuple(range(8))),
                        dates, f"cnn_{stage}")
        area = sum(row["mapped_wet_area_km2"] for row in rows)
        summary = {key: round(sum(row[key] * row["mapped_wet_area_km2"]
                                  for row in rows) / area, 5)
                   for key in ("top_5pct_area_capture", "top_10pct_area_capture",
                               "top_20pct_area_capture")}
        report["stages"][stage] = {
            "training_patches": train_regions,
            "added_2025_dates": dates[8:],
            "positive_training_patch_days": training["train_patch_days"],
            "positive_validation_patch_days": training["val_patch_days"],
            "normalization": stats, "training": training,
            "bangkok_summary": summary, "bangkok_dates": rows}
        DEST.write_text(json.dumps(report, indent=2) + "\n")
        print("STAGE", stage, len(train_regions), summary, flush=True)


if __name__ == "__main__":
    main()
