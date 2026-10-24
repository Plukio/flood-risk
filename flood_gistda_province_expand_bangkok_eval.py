"""Evaluate province-trained CNNs on GISTDA water inside Bangkok's boundary."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_bangkok_gistda_grid_backtest import selection_weights
from flood_bangkok_transfer import load_districts
from flood_bangkapi_surface_grid import GRID_M, grid_spec
from flood_gistda_multiregion_data import OUTPUT, SIZE, specs
from flood_gistda_multiregion_train import SpatialRankCNN
from flood_gistda_province_expand_train import FEATURE_NAMES, TARGET, load_region


SOURCE = OUTPUT / "province_expansion_bangkok_metrics.json"
AUGMENTED = OUTPUT / "province_expansion_2025_augmented_bangkok_metrics.json"
DEST = OUTPUT / "province_expansion_bangkok_bma_only.json"
GRID_OUTPUT = OUTPUT.parent / "flood_bangkapi_surface"
FRACTIONS = (.05, .10, .20)


def bma_arrays(dates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    bounds = specs()[TARGET]["utm_bounds"]
    west, _, _, north = bounds
    districts = {d["code"]: grid_spec(d["polygon"]) for d in load_districts()}
    area = np.zeros((SIZE, SIZE), np.float64)
    wet = np.zeros((len(dates), SIZE, SIZE), np.float64)
    for di, day in enumerate(dates):
        path = GRID_OUTPUT / f"gistda_{day.replace('-', '')}_grid_250m.csv"
        current_area = np.zeros_like(area)
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                grid = districts[row["district_code"]]
                x = grid["west"] + int(row["col"]) * GRID_M
                y = grid["north"] - int(row["row"]) * GRID_M
                col = round((x - west) / GRID_M)
                rr = round((north - y) / GRID_M)
                if 0 <= col < SIZE and 0 <= rr < SIZE:
                    current_area[rr, col] += float(row["valid_cell_area_m2"])
                    wet[di, rr, col] += float(row["mapped_wet_area_m2"])
        if np.any(current_area > GRID_M**2 * 1.02):
            raise ValueError(f"BMA district cell overlap on {day}")
        if di == 0:
            area = current_area
        elif np.max(abs(area - current_area)) > 1:
            raise ValueError("BMA candidate cell areas changed between dates")
    if np.any(wet > area[None] + 2):
        raise ValueError("Mapped water exceeds BMA cell area")
    return area, wet


def metric(score: np.ndarray, wet: np.ndarray, mask: np.ndarray) -> dict:
    positives = wet[mask]
    if positives.sum() <= 0:
        raise ValueError("No BMA mapped water in target patch")
    return {f"top_{int(f*100)}pct_capture": round(float(np.dot(
        selection_weights(score[mask], f), positives) / positives.sum()), 5)
            for f in FRACTIONS}


def main() -> None:
    torch.set_num_threads(4)
    source = json.loads(SOURCE.read_text())
    if AUGMENTED.exists():
        source["stages"].update(json.loads(AUGMENTED.read_text())["stages"])
    region, dates = load_region(TARGET)
    area, wet = bma_arrays(dates)
    mask = area > 1000
    if not mask.any():
        raise ValueError("No Bangkok cells in patch")
    patch_wet = region["wet"].sum(axis=(1, 2))
    if np.any(wet.sum(axis=(1, 2)) > patch_wet * 1.01):
        raise ValueError("BMA subset exceeds whole patch positive area")
    scores = {"crop_fraction": np.broadcast_to(region["x"][0, 17],
              (len(dates), SIZE, SIZE))}
    for stage, info in source["stages"].items():
        checkpoint = torch.load(
            OUTPUT / f"gistda_expanded_bkk_{stage}_cnn_spatial_rank.pt",
            map_location="cpu", weights_only=False)
        if TARGET in checkpoint["train_regions"] or tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Bangkok target leaked into checkpoint")
        mean = np.asarray(info["normalization"]["mean"], np.float32)
        std = np.asarray(info["normalization"]["std"], np.float32)
        x = np.clip((region["x"] - mean[None, :, None, None]) /
                    std[None, :, None, None], -5, 5).astype(np.float32)
        model = SpatialRankCNN(len(FEATURE_NAMES))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            scores[stage] = model(torch.from_numpy(x)).numpy()
    output = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "target": "GISTDA mapped-positive area inside BMA boundary and fixed Nong Chok patch",
              "candidate_bma_cells": int(mask.sum()),
              "patch_cells": SIZE*SIZE,
              "bma_area_km2": round(float(area.sum())/1e6, 4),
              "model_results": {}}
    for name, score in scores.items():
        rows = []
        for i, day in enumerate(dates):
            if wet[i].sum() <= 0:
                continue
            rows.append({"date": day,
                         "bma_mapped_area_km2": round(float(wet[i].sum())/1e6, 5),
                         "whole_patch_mapped_area_km2": round(float(patch_wet[i])/1e6, 5),
                         **metric(score[i], wet[i], mask)})
        total = sum(row["bma_mapped_area_km2"] for row in rows)
        summary = {f"top_{int(f*100)}pct_capture": round(sum(
            row[f"top_{int(f*100)}pct_capture"] * row["bma_mapped_area_km2"]
            for row in rows) / total, 5) for f in FRACTIONS}
        output["model_results"][name] = {"summary": summary, "dates": rows}
        print(name, summary, flush=True)
    DEST.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
