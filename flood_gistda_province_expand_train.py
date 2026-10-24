"""Measure whether more province patches improve a fixed Bangkok holdout.

Bangkok's Nong Chok patch is never used for weight fitting, normalization,
or epoch selection. All stages share its eight GISTDA-positive evaluation days.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_gistda_multiregion_data import CACHE, CENTERS, OUTPUT, SIZE
from flood_gistda_multiregion_train import (
    BASE_FEATURE_NAMES, evaluate, normalize, scenes, train_one,
)
from flood_gistda_multiregion_worldcover import CHANNELS
from flood_gistda_province_expand_data import selected_specs


TARGET = "nong_chok"
FIRST_FIVE = ("suphan_buri_city", "ang_thong", "prachin_buri_city",
              "phichit", "sukhothai")
NEXT_FIVE = ("chainat", "sing_buri", "lop_buri", "nakhon_nayok",
             "nakhon_pathom")
FINAL_FIVE = ("suphan_buri_north", "prachin_buri_east", "ratchaburi",
              "phetchaburi", "nakhon_sawan_east")
STAGES = {
    "original_5": (),
    "plus_5": FIRST_FIVE,
    "plus_10": FIRST_FIVE + NEXT_FIVE,
    "plus_15": FIRST_FIVE + NEXT_FIVE + FINAL_FIVE,
}
FEATURE_NAMES = BASE_FEATURE_NAMES + CHANNELS


def load_region(name: str) -> tuple[dict, list[str]]:
    with np.load(CACHE / f"{name}_wet_250m.npz") as saved:
        wet = saved["wet_area_m2"].astype(np.float32)
        dates = list(saved["dates"].astype(str))
    with np.load(CACHE / f"{name}_static_250m.npz") as saved:
        static = saved["values"].astype(np.float32)
    with np.load(CACHE / f"{name}_rain_features_2026.npz") as saved:
        rain = saved["values"].astype(np.float32)
        if list(saved["dates"].astype(str)) != dates:
            raise ValueError(f"Rain dates mismatch: {name}")
    with np.load(CACHE / f"{name}_worldcover_2021_250m.npz") as saved:
        cover = saved["values"].astype(np.float32)
        if tuple(saved["names"].astype(str)) != CHANNELS:
            raise ValueError(f"WorldCover channel order mismatch: {name}")
    if (wet.shape != (8, SIZE, SIZE) or static.shape != (7, SIZE, SIZE)
            or rain.shape != (8, 7) or cover.shape != (6, SIZE, SIZE)):
        raise ValueError(f"Input shape mismatch: {name}")
    month = np.asarray([int(day[5:7]) for day in dates])
    global_values = np.concatenate((rain,
        np.sin(2 * np.pi * month[:, None] / 12),
        np.cos(2 * np.pi * month[:, None] / 12)), axis=1)
    x = np.concatenate((
        np.broadcast_to(static, (8, 7, SIZE, SIZE)),
        np.broadcast_to(global_values[:, :, None, None], (8, 9, SIZE, SIZE)),
        np.broadcast_to(cover, (8, 6, SIZE, SIZE))), axis=1).copy()
    return {"x": x, "wet": wet, "static": static}, dates


def load_all() -> tuple[dict, list[str]]:
    names = list(CENTERS) + list(selected_specs())
    regions = {}
    dates = None
    for name in names:
        regions[name], current = load_region(name)
        if dates is None:
            dates = current
        elif dates != current:
            raise ValueError(f"Date mismatch: {name}")
    return regions, dates


def main() -> None:
    torch.set_num_threads(4)
    selected = tuple(selected_specs())
    if set(FIRST_FIVE + NEXT_FIVE + FINAL_FIVE) != set(selected):
        raise ValueError("Stage lists do not cover selected province patches")
    old_external = tuple(name for name in CENTERS if name != TARGET)
    report = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "target": "GISTDA mapped-water positive area in held-out Nong Chok 20 km patch",
              "not_target": "Verified dry/wet flood probability or road flood",
              "feature_names": FEATURE_NAMES,
              "source": "2021 ESA WorldCover; 2026 GISTDA polygons; SRTM; 2025 OSM; Open-Meteo rain",
              "bangkok_holdout": TARGET, "stages": {}}
    for stage, extra in STAGES.items():
        train_regions = old_external + extra
        regions, dates = load_all()
        normalizer = normalize(regions, train_regions)
        model, training = train_one("cnn", regions, dates, train_regions,
                                    FEATURE_NAMES, f"expanded_bkk_{stage}")
        pairs = scenes(regions, (TARGET,), tuple(range(len(dates))))
        rows = evaluate(model, regions, pairs, dates, f"cnn_{stage}")
        area = sum(row["mapped_wet_area_km2"] for row in rows)
        summary = {key: round(sum(row[key] * row["mapped_wet_area_km2"]
                                  for row in rows) / area, 5)
                   for key in ("top_5pct_area_capture", "top_10pct_area_capture",
                               "top_20pct_area_capture")}
        report["stages"][stage] = {
            "training_patches": train_regions, "added_patches": extra,
            "positive_training_patch_days": training["train_patch_days"],
            "positive_validation_patch_days": training["val_patch_days"],
            "normalization": normalizer, "training": training,
            "bangkok_summary": summary, "bangkok_dates": rows}
        (OUTPUT / "province_expansion_bangkok_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n")
        print("STAGE", stage, len(train_regions), summary, flush=True)


if __name__ == "__main__":
    main()
