"""Compare Bangkok-held-out province CNNs with the new U-Net on identical targets."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone

import numpy as np
import torch
from shapely.geometry import box
from shapely.ops import unary_union

from flood_bangkok_transfer import load_districts
from flood_gistda_multiregion_data import CACHE as STATIC_CACHE, GRID_M, OUTPUT as CNN_OUTPUT
from flood_gistda_multiregion_train import SpatialRankCNN, STATIC_NAMES, RAIN_NAMES
from flood_gistda_multiregion_worldcover import CHANNELS as COVER_NAMES
from flood_gistda_province_expand_bangkok_eval import bma_arrays
from flood_gistda_province_expand_train import FEATURE_NAMES as CNN_FEATURE_NAMES
from flood_gistda_province_expand_traffy_stress import reports as traffy_reports
from flood_gistda_province_expand_traffy_stress import weather as traffy_weather
from flood_open_weak_bangkok_holdout_eval import aggregate, bma_fraction, score_set, selected
from flood_open_weak_data import FEATURE_NAMES, OUTPUT, ROOT, all_specs


STAGES = ("plus_10", "plus_10_plus_2025")
OUT = OUTPUT / "bangkok_holdout" / "cnn_comparison.json"


def load_model(stage: str, bma, specs: dict):
    report_name = ("province_expansion_2025_augmented_bangkok_metrics.json"
                   if "plus_2025" in stage else "province_expansion_bangkok_metrics.json")
    report = json.loads((CNN_OUTPUT / report_name).read_text())
    info = report["stages"][stage]
    checkpoint_path = CNN_OUTPUT / f"gistda_expanded_bkk_{stage}_cnn_spatial_rank.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["feature_names"]) != CNN_FEATURE_NAMES:
        raise ValueError("CNN feature names differ")
    if set(checkpoint["train_regions"]) != set(info["training_patches"]):
        raise ValueError("CNN training-region metadata differs")
    if any(bma.intersection(box(*specs[r]["utm_bounds"])).area > 0
           for r in checkpoint["train_regions"]):
        raise ValueError("CNN has a Bangkok-overlapping training patch")
    model = SpatialRankCNN(len(CNN_FEATURE_NAMES))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    norm = info["normalization"]
    return model, norm, checkpoint_path, tuple(checkpoint["train_regions"])


def predict(model, norm, static: np.ndarray, weather: np.ndarray) -> np.ndarray:
    x = np.concatenate((static[:7],
                        np.broadcast_to(weather[:, None, None], (9, 80, 80)),
                        static[7:]), axis=0).astype(np.float32)
    mean = np.asarray(norm["mean"], np.float32)[:, None, None]
    std = np.asarray(norm["std"], np.float32)[:, None, None]
    x = np.clip((x - mean) / std, -5, 5)
    with torch.no_grad():
        return model(torch.from_numpy(x[None]))[0].numpy()


def main() -> None:
    torch.set_num_threads(4)
    if FEATURE_NAMES[:7] != STATIC_NAMES or FEATURE_NAMES[7:] != COVER_NAMES:
        raise ValueError("CNN/U-Net static feature order differs")
    if CNN_FEATURE_NAMES != STATIC_NAMES + RAIN_NAMES + ("month_sin", "month_cos") + COVER_NAMES:
        raise ValueError("CNN feature layout differs")
    manifest = json.loads((OUTPUT / "data_audit.json").read_text())
    eos = {s["region"]: s for s in manifest["scenes"] if s["source"] == "eos2026"}
    specs = all_specs()
    bma = unary_union([d["polygon"] for d in load_districts()])
    with np.load(STATIC_CACHE / "nong_chok_rain_features_2026.npz") as z:
        rain, days = z["values"].astype(np.float32), z["dates"].astype(str)
    _, wet_bma = bma_arrays(list(days))
    history, reports, report_days = traffy_reports()
    report_weather = traffy_weather(report_days)
    output = {"evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "metric": "positive-area or positive-report capture in top 10% of the same candidate cells as U-Net",
              "stages": {}}
    for stage in STAGES:
        model, norm, checkpoint, regions = load_model(stage, bma, specs)
        result = {"checkpoint": str(checkpoint.relative_to(ROOT)),
                  "training_regions": list(regions), "eos_bangkok_2026_09_27": {},
                  "gistda_nong_chok_2026": []}
        for region in ("bang_kapi_core", "nong_chok"):
            with np.load(ROOT / eos[region]["path"]) as z:
                static, weather, wet = (z[k].astype(np.float32) for k in ("static", "weather", "wet"))
            area = bma_fraction(specs[region], bma)
            score = predict(model, norm, static, weather)
            result["eos_bangkok_2026_09_27"][region] = score_set(
                score, static, wet * area, area >= .5, "cnn")
        with np.load(ROOT / eos["nong_chok"]["path"]) as z:
            static = z["static"].astype(np.float32)
        area = bma_fraction(specs["nong_chok"], bma)
        for i, stamp in enumerate(days):
            month = date.fromisoformat(stamp).month
            weather = np.concatenate((rain[i], [math.sin(2*math.pi*month/12),
                                               math.cos(2*math.pi*month/12)])).astype(np.float32)
            score = predict(model, norm, static, weather)
            row = score_set(score, static, wet_bma[i].astype(np.float32) / GRID_M**2,
                            area >= .5, "cnn")
            result["gistda_nong_chok_2026"].append({"day": stamp, "results": row})
        result["summary"] = {
            "eos_patch_weighted": aggregate(list(result["eos_bangkok_2026_09_27"].values())),
            "gistda_date_weighted": aggregate([r["results"] for r in result["gistda_nong_chok_2026"]])}
        mask = area > 1000 / GRID_M**2
        scores = np.stack([predict(model, norm, static, w) for w in report_weather])
        hit = sum(np.dot(selected(s[mask], .10), y[mask])
                  for s, y in zip(scores, reports))
        result["traffy_2025_q4_stress"] = {
            "report_positive_cell_days": int(reports[:, mask].sum()),
            "candidate_cells": int(mask.sum()),
            "top_10pct_report_capture": round(float(hit / reports[:, mask].sum()), 5)}
        output["stages"][stage] = result
        print(stage, "EOS", result["summary"]["eos_patch_weighted"]["all_bma"]["cnn"]["area_weighted_capture"]["top_10pct"],
              "GISTDA", result["summary"]["gistda_date_weighted"]["all_bma"]["cnn"]["area_weighted_capture"]["top_10pct"],
              "Traffy", result["traffy_2025_q4_stress"]["top_10pct_report_capture"], flush=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
