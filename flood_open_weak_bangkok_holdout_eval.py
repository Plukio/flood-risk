"""Evaluate a Bangkok-excluded U-Net or CNN on unseen Bangkok proxy maps."""

from __future__ import annotations

import json
import math
import argparse
from datetime import date, datetime, timezone

import numpy as np
import torch
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.ops import unary_union
from scipy.stats import spearmanr

from flood_bangkok_transfer import load_districts
from flood_gistda_multiregion_data import CACHE as STATIC_CACHE, GRID_M, SIZE
from flood_gistda_province_expand_bangkok_eval import bma_arrays
from flood_gistda_province_expand_traffy_stress import reports as traffy_reports
from flood_gistda_province_expand_traffy_stress import weather as traffy_weather
from flood_open_weak_data import FEATURE_NAMES, WEATHER_NAMES, OUTPUT, ROOT, all_specs
from flood_open_weak_train import WeatherUNet, WeatherCNN


FRACTIONS = (.05, .10, .20)


def selected(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Allocate the same fraction of cells, splitting score ties evenly."""
    _, inverse, counts = np.unique(-np.asarray(scores, dtype=np.float64), return_inverse=True,
                                    return_counts=True)
    before = np.cumsum(counts) - counts
    share = np.clip((fraction * len(scores) - before) / counts, 0, 1)
    return share[inverse]


def capture(scores: np.ndarray, wet: np.ndarray, mask: np.ndarray) -> dict:
    truth = wet[mask]
    if truth.sum() <= 0:
        return {"candidate_cells": int(mask.sum()), "mapped_km2": 0,
                "capture": None}
    values = scores[mask]
    return {"candidate_cells": int(mask.sum()),
            "mapped_km2": round(float(truth.sum()) * GRID_M**2 / 1e6, 4),
            "capture": {f"top_{int(f*100)}pct": round(float(np.dot(
                selected(values, f), truth) / truth.sum()), 5)
                for f in FRACTIONS}}


def bma_fraction(spec: dict, polygon) -> np.ndarray:
    west, _, _, north = spec["utm_bounds"]
    small = rasterize([(polygon, 1)], out_shape=(SIZE*5, SIZE*5),
                      transform=from_origin(west, north, GRID_M/5, GRID_M/5),
                      dtype="uint8")
    return small.reshape(SIZE, 5, SIZE, 5).mean(axis=(1, 3))


def normalize(static: np.ndarray, weather: np.ndarray, norm: dict):
    mean = np.asarray(norm["static_mean"], np.float32)[:, None, None]
    std = np.asarray(norm["static_std"], np.float32)[:, None, None]
    wmean = np.asarray(norm["weather_mean"], np.float32)
    wstd = np.asarray(norm["weather_std"], np.float32)
    return (np.clip((static-mean)/std, -5, 5).astype(np.float32),
            np.clip((weather-wmean)/wstd, -5, 5).astype(np.float32))


def model_scores(model, norm, static, weather):
    x, w = normalize(static, weather, norm)
    with torch.no_grad():
        return model(torch.from_numpy(x[None]), torch.from_numpy(w[None]))[0].numpy()


def score_set(score, static, wet, mask, model_name="unet"):
    built = static[FEATURE_NAMES.index("worldcover_built_fraction_2021")]
    crop = static[FEATURE_NAMES.index("worldcover_crop_fraction_2021")]
    elevation = static[FEATURE_NAMES.index("srtm_elevation_m")]
    strata = {"all_bma": mask, "built_gt_20pct": mask & (built > .20),
              "crop_gt_50pct": mask & (crop > .50)}
    return {name: {model_name: capture(score, wet, subset),
                   "crop_fraction": capture(crop, wet, subset),
                   "built_fraction": capture(built, wet, subset),
                   "low_elevation": capture(-elevation, wet, subset)}
            for name, subset in strata.items() if subset.any()}


def aggregate(results: list[dict]) -> dict:
    """Area-weighted capture across patches or dates; areas may repeat by date."""
    answer = {}
    for stratum in results[0]:
        answer[stratum] = {}
        for method in results[0][stratum]:
            rows = [r[stratum][method] for r in results]
            total = sum(r["mapped_km2"] for r in rows)
            answer[stratum][method] = {
                "mapped_km2_sum": round(total, 4),
                "area_weighted_capture": {
                    f"top_{int(f*100)}pct": round(sum(
                        r["mapped_km2"] * r["capture"][f"top_{int(f*100)}pct"]
                        for r in rows if r["capture"]) / total, 5)
                    for f in FRACTIONS} if total else None}
    return answer


def traffy_stress(model, norm, static, mask, model_name="unet") -> dict:
    history, truth, days = traffy_reports()
    if int(truth[:, mask].sum()) < 20:
        raise ValueError("Too few Bangkok report-positive cell-days")
    weather = traffy_weather(days)
    scores = np.stack([model_scores(model, norm, static, w) for w in weather])
    built = static[FEATURE_NAMES.index("worldcover_built_fraction_2021")]
    crop = static[FEATURE_NAMES.index("worldcover_crop_fraction_2021")]
    outcome = {"target": "2025 Q4 Traffy flood-report-positive cell-days in Nong Chok BMA patch",
               "chronology": "2026-trained model applied backward to 2025; cross-target stress only",
               "candidate_cells": int(mask.sum()),
               "report_positive_cell_days": int(truth[:, mask].sum()),
               "report_positive_days": int((truth[:, mask].sum(axis=1) > 0).sum()),
               "top_10pct_report_capture": {}}
    for name, value in {model_name: scores, "2025_report_history": history,
                        "built_fraction": built, "crop_fraction": crop}.items():
        if value.ndim == 2:
            picks = selected(value[mask], .10)
            captured = float((truth[:, mask] * picks[None]).sum() / truth[:, mask].sum())
        else:
            captured = float(sum(np.dot(selected(s[mask], .10), y[mask])
                                 for s, y in zip(value, truth)) / truth[:, mask].sum())
        outcome["top_10pct_report_capture"][name] = round(captured, 5)
    outcome["cell_spearman_report_days"] = round(float(spearmanr(
        scores[:, mask].mean(axis=0), truth[:, mask].sum(axis=0)).statistic), 5)
    return outcome


def main(model_name: str = "unet") -> None:
    torch.set_num_threads(4)
    out = OUTPUT / ("bangkok_holdout_cnn" if model_name == "cnn" else "bangkok_holdout")
    checkpoint_path = out / f"weather_{model_name}_bangkok_holdout.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    excluded = {"nong_chok", "bang_kapi_core"}
    if set(checkpoint["excluded_regions"]) != excluded:
        raise ValueError("Checkpoint did not exclude both Bangkok regions")
    if tuple(checkpoint["feature_names"]) != FEATURE_NAMES or tuple(checkpoint["weather_names"]) != WEATHER_NAMES:
        raise ValueError("Feature order changed")
    model_class = {"unet": WeatherUNet, "cnn": WeatherCNN}[model_name]
    if checkpoint["model"] != model_class.__name__:
        raise ValueError("Checkpoint architecture differs")
    manifest = json.loads((OUTPUT / "data_audit.json").read_text())
    train_regions = {s["region"] for s in manifest["scenes"] if s["region"] not in excluded}
    bma = unary_union([d["polygon"] for d in load_districts()])
    from shapely.geometry import box
    overlaps = {region: bma.intersection(box(*all_specs()[region]["utm_bounds"])).area
                for region in train_regions}
    if any(area > 0 for area in overlaps.values()):
        raise ValueError(f"Training patch overlaps Bangkok: {overlaps}")
    model = model_class(len(FEATURE_NAMES), len(WEATHER_NAMES))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output = {"evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "checkpoint": str(checkpoint_path.relative_to(ROOT)),
              "model": model_class.__name__,
              "training_regions": sorted(train_regions),
              "excluded_regions": sorted(excluded),
              "metric": "fraction of mapped positive area in top-ranked candidate cells; unknown cells are not dry",
              "eos_bangkok_2026_09_27": {}, "gistda_nong_chok_2026": []}
    specs = all_specs()
    eos = {s["region"]: s for s in manifest["scenes"] if s["source"] == "eos2026"}
    for region in sorted(excluded):
        with np.load(ROOT / eos[region]["path"]) as z:
            static, weather, wet = (z[k].astype(np.float32) for k in ("static", "weather", "wet"))
        spec = specs[region]
        area = bma_fraction(spec, bma)
        mask = area >= .5
        score = model_scores(model, checkpoint["normalization"], static, weather)
        output["eos_bangkok_2026_09_27"][region] = score_set(
            score, static, wet * area, mask, model_name)

    # GISTDA dates are additional positive-only proxy maps in the Nong Chok
    # patch. The 27 Sep map may share satellite observations with EOS-RS.
    with np.load(STATIC_CACHE / "nong_chok_rain_features_2026.npz") as z:
        rain, days = z["values"].astype(np.float32), z["dates"].astype(str)
    _, wet_bma = bma_arrays(list(days))
    with np.load(ROOT / eos["nong_chok"]["path"]) as z:
        static = z["static"].astype(np.float32)
    area = bma_fraction(specs["nong_chok"], bma)
    mask = area >= .5
    for i, stamp in enumerate(days):
        month = date.fromisoformat(stamp).month
        weather = np.concatenate((rain[i], [math.sin(2*math.pi*month/12),
                                            math.cos(2*math.pi*month/12)])).astype(np.float32)
        score = model_scores(model, checkpoint["normalization"], static, weather)
        result = score_set(score, static, wet_bma[i].astype(np.float32) / GRID_M**2,
                           mask, model_name)
        output["gistda_nong_chok_2026"].append({"day": stamp, "results": result})
    output["summary"] = {
        "eos_patch_weighted": aggregate(list(output["eos_bangkok_2026_09_27"].values())),
        "gistda_date_weighted": aggregate([r["results"] for r in output["gistda_nong_chok_2026"]])}
    output["traffy_2025_q4_stress"] = traffy_stress(
        model, checkpoint["normalization"], static,
        area > 1000/GRID_M**2, model_name)
    (out / "bangkok_holdout_eval.json").write_text(json.dumps(output, indent=2) + "\n")
    for region, results in output["eos_bangkok_2026_09_27"].items():
        print("EOS", region, {s: {m: v["capture"]["top_10pct"] if v["capture"] else None
                                 for m, v in metrics.items()}
                              for s, metrics in results.items()})
    for row in output["gistda_nong_chok_2026"]:
        print("GISTDA", row["day"], {m: v["capture"]["top_10pct"] if v["capture"] else None
                                      for m, v in row["results"]["all_bma"].items()})
    print("TRAFFY", output["traffy_2025_q4_stress"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("unet", "cnn"), default="unet")
    main(parser.parse_args().model)
