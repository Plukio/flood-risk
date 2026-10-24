"""Cross-target stress check: province water models against Bangkok reports.

The models were fitted on 2026 GISTDA water and are applied backward to 2025
Traffy reports. This diagnoses target mismatch, not a forward-time forecast.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests
import torch
from scipy.special import softmax
from scipy.stats import spearmanr

from flood_bangkok_gistda_grid_backtest import selection_weights
from flood_bangkok_spatial_cnn_citywide import FEATURE_CACHE
from flood_bangkok_transfer import RAIN_URL, load_districts
from flood_bangkapi_surface_grid import GRID_M, grid_spec
from flood_bangkapi_surface_model import RAIN_FEATURES, rainfall_by_date
from flood_gistda_multiregion_data import CACHE, OUTPUT, SIZE, specs
from flood_gistda_multiregion_train import SpatialRankCNN
from flood_gistda_province_expand_bangkok_eval import bma_arrays
from flood_gistda_province_expand_train import FEATURE_NAMES, TARGET, load_region


START = date(2025, 10, 1)
END = date(2025, 12, 31)
RAIN_PATH = CACHE / "nong_chok_patch_2025_q4_rain.json"
DEST = OUTPUT / "province_expansion_traffy_2025_stress.json"


def reports() -> tuple[np.ndarray, np.ndarray, list[date]]:
    target = specs()[TARGET]["utm_bounds"]
    west, _, _, north = target
    days = [START + timedelta(days=i) for i in range((END-START).days+1)]
    proxy = np.zeros((365, SIZE, SIZE), np.uint8)
    contributing = []
    for district in load_districts():
        grid = grid_spec(district["polygon"])
        if (grid["east"] <= west or grid["west"] >= target[2] or
                grid["north"] <= target[1] or grid["south"] >= north):
            continue
        code = district["code"]
        y_path = FEATURE_CACHE / f"{code}_y.npy"
        if not y_path.exists():
            raise FileNotFoundError(y_path)
        y = np.load(y_path, mmap_mode="r")
        if len(y) != 1461:
            raise ValueError(f"Report time length changed: {code}")
        positive = np.argwhere(y[1096:1461, 0] > 0)
        for di, rr, cc in positive:
            col = round((grid["west"] + int(cc)*GRID_M - west) / GRID_M)
            row = round((north - (grid["north"] - int(rr)*GRID_M)) / GRID_M)
            if 0 <= col < SIZE and 0 <= row < SIZE:
                proxy[int(di), row, col] = 1
        contributing.append(code)
    return proxy[:273].sum(axis=0), proxy[273:], days


def weather(days: list[date]) -> np.ndarray:
    if not RAIN_PATH.exists():
        lon, lat = specs()[TARGET]["center_lonlat"]
        response = requests.get(RAIN_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(START - timedelta(days=3)),
            "end_date": str(END), "hourly": "precipitation",
            "timezone": "Asia/Bangkok"}, timeout=70)
        if response.status_code != 200:
            raise RuntimeError(f"Open-Meteo returned HTTP {response.status_code}")
        RAIN_PATH.write_text(response.text)
    values = rainfall_by_date(START, END, RAIN_PATH)
    month = np.asarray([d.month for d in days])
    rain = np.asarray([[math.log1p(max(0, values[d][key]))
                       for key in RAIN_FEATURES] for d in days], np.float32)
    return np.concatenate((rain,
        np.sin(2*np.pi*month[:,None]/12),
        np.cos(2*np.pi*month[:,None]/12)), axis=1).astype(np.float32)


def inputs(days: list[date]) -> np.ndarray:
    item, _ = load_region(TARGET)
    static = item["static"]
    cover = item["x"][0, 16:]
    globals_ = weather(days)
    return np.concatenate((
        np.broadcast_to(static, (len(days), 7, SIZE, SIZE)),
        np.broadcast_to(globals_[:, :, None, None],
                        (len(days), 9, SIZE, SIZE)),
        np.broadcast_to(cover, (len(days), 6, SIZE, SIZE))), axis=1).astype(np.float32)


def metrics(score: np.ndarray, truth: np.ndarray,
            mask: np.ndarray, daily: bool) -> dict:
    if score.ndim == 2:
        score = np.broadcast_to(score, truth.shape)
    selected = np.asarray([selection_weights(s[mask], .10)
                           for s in score])
    target = truth[:, mask]
    capture = float((selected * target).sum() / target.sum())
    seasonal = (np.asarray([softmax(s[mask]) for s in score]).sum(axis=0)
                if daily else score[0, mask])
    correlation = spearmanr(seasonal, target.sum(axis=0)).statistic
    return {"top_10pct_report_positive_capture": round(capture, 5),
            "cell_spearman_report_days": round(float(correlation), 5)}


def main() -> None:
    torch.set_num_threads(4)
    _, dates_2026 = load_region(TARGET)
    bma_area, _ = bma_arrays(dates_2026)
    mask = bma_area > 1000
    history, truth, days = reports()
    if truth.sum() < 20:
        raise ValueError("Too few Bangkok report-positive cell-days")
    x = inputs(days)
    source = json.loads((OUTPUT / "province_expansion_bangkok_metrics.json").read_text())
    augmented = OUTPUT / "province_expansion_2025_augmented_bangkok_metrics.json"
    if augmented.exists():
        source["stages"].update(json.loads(augmented.read_text())["stages"])
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "target": "2025 Q4 Traffy flood-report-positive 250 m cell-days inside BMA boundary and Nong Chok patch",
              "chronology": "2026-trained models applied backward to 2025; cross-target diagnostic, not forward-time validation",
              "report_positive_cell_days": int(truth[:,mask].sum()),
              "report_positive_days": int((truth[:,mask].sum(axis=1)>0).sum()),
              "candidate_bma_cells": int(mask.sum()),
              "models": {}}
    cover = x[0, 16:]
    for name, score in (("2025_report_history", history),
                        ("built_fraction", cover[0]),
                        ("crop_fraction", cover[1])):
        result["models"][name] = metrics(score, truth, mask, False)
    for stage, info in source["stages"].items():
        checkpoint = torch.load(
            OUTPUT / f"gistda_expanded_bkk_{stage}_cnn_spatial_rank.pt",
            map_location="cpu", weights_only=False)
        if TARGET in checkpoint["train_regions"] or tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Bangkok target leaked into model")
        mean = np.asarray(info["normalization"]["mean"], np.float32)
        std = np.asarray(info["normalization"]["std"], np.float32)
        normalized = np.clip((x - mean[None,:,None,None]) /
                             std[None,:,None,None], -5, 5)
        model = SpatialRankCNN(len(FEATURE_NAMES))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(days), 8):
                scores.append(model(torch.from_numpy(
                    normalized[start:start+8])).numpy())
        score = np.concatenate(scores)
        result["models"][stage] = metrics(score, truth, mask, True)
        print(stage, result["models"][stage], flush=True)
    DEST.write_text(json.dumps(result, indent=2) + "\n")
    print("baselines", {name: result["models"][name] for name in
                         ("2025_report_history", "built_fraction", "crop_fraction")}, flush=True)


if __name__ == "__main__":
    main()
