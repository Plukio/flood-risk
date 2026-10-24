"""Simple land-cover ranking checks for the six-region mapped-water target."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np

from flood_gistda_multiregion_2025_test import PAIRS
from flood_gistda_multiregion_data import CACHE, CENTERS, OUTPUT
from flood_gistda_multiregion_train import capture
from flood_gistda_multiregion_worldcover import CHANNELS


RANKS = {
    "crop_fraction": lambda x: x[1],
    "low_built_fraction": lambda x: -x[0],
    "wetland_fraction": lambda x: x[4],
    "water_fraction": lambda x: x[5],
}


def cover(name: str) -> np.ndarray:
    with np.load(CACHE / f"{name}_worldcover_2021_250m.npz") as saved:
        if tuple(saved["names"].astype(str)) != CHANNELS:
            raise ValueError(f"WorldCover channels changed: {name}")
        return saved["values"].astype(np.float32)


def add(rows: list[dict], split: str, region: str, date: str,
        features: np.ndarray, wet: np.ndarray) -> None:
    area = float(wet.sum()) / 1e6
    if area <= 0:
        return
    for model, score in RANKS.items():
        rows.append({"split": split, "region": region, "date": date,
                     "model": model, "mapped_wet_area_km2": area,
                     **capture(score(features), wet)})


def aggregate(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["split"], row["model"]), []).append(row)
    output = []
    for (split, model), group in groups.items():
        area = sum(row["mapped_wet_area_km2"] for row in group)
        output.append({"split": split, "model": model,
                       "positive_patch_days": len(group),
                       "mapped_area_days_km2": round(area, 4),
                       **{f"top_{pct}pct_area_capture": round(sum(
                           row[f"top_{pct}pct_area_capture"] * row["mapped_wet_area_km2"]
                           for row in group) / area, 5) for pct in (5, 10, 20)}})
    return sorted(output, key=lambda row: (row["split"], row["model"]))


def main() -> None:
    rows = []
    mapped_cover = np.zeros(len(CHANNELS), dtype=np.float64)
    mapped_area = 0.0
    patch_cover = np.zeros(len(CHANNELS), dtype=np.float64)
    mapped_thresholds = {"crop_over_50pct_cell": 0.0,
                         "built_over_20pct_cell": 0.0}
    for region in CENTERS:
        x = cover(region)
        patch_cover += x.mean(axis=(1, 2)) / len(CENTERS)
        with np.load(CACHE / f"{region}_wet_250m.npz") as saved:
            wet = saved["wet_area_m2"].astype(np.float32)
            dates = saved["dates"].astype(str)
        summed_wet = wet.sum(axis=0)
        mapped_cover += (x * summed_wet).sum(axis=(1, 2))
        mapped_area += float(summed_wet.sum())
        mapped_thresholds["crop_over_50pct_cell"] += float(summed_wet[x[1] > .5].sum())
        mapped_thresholds["built_over_20pct_cell"] += float(summed_wet[x[0] > .2].sum())
        for date, area in zip(dates, wet):
            add(rows, "2026_geographic", region, date, x, area)
    for region, date in PAIRS:
        with np.load(CACHE / f"{region}_wet_{date}_250m.npz") as saved:
            wet = saved["wet_area_m2"].astype(np.float32)
        add(rows, "2025_selected_maps", region, str(date), cover(region), wet)
    summary = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "source": "ESA WorldCover 2021",
               "meaning": "Untrained static rankings of GISTDA positive mapped-water area; not flood probability",
               "estimated_2026_cover_composition": {
                   "method": "250 m WorldCover class fractions weighted by GISTDA mapped-water area in each cell; approximate, not 10 m polygon overlay",
                   "mapped_positive": {name: round(float(mapped_cover[i] / mapped_area), 5)
                                       for i, name in enumerate(CHANNELS)},
                   "six_patch_mean": {name: round(float(patch_cover[i]), 5)
                                      for i, name in enumerate(CHANNELS)},
                   "mapped_positive_in_cell_types": {
                       name: round(area / mapped_area, 5)
                       for name, area in mapped_thresholds.items()}},
               "summary": aggregate(rows), "rows": rows}
    (OUTPUT / "gistda_cover_simple_baselines.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    with (OUTPUT / "gistda_cover_simple_baselines.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in summary["summary"]:
        print(row)


if __name__ == "__main__":
    main()
