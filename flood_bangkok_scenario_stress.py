"""Rain-input stress test for saved Bangkok complaint CNN and CNN–GNN.

This changes only rainfall summaries on a fixed 1 Oct 2025 scene. It is a
model response audit, not a hydraulic simulation or predicted flood map.
"""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

from flood_bangkok_cnn_gnn_hybrid import FlowHybrid, load_graphs
from flood_bangkok_spatial_cnn_citywide import (
    CAL_N, CHANNELS, HOLDOUT, prepare_cache, transform,
)
from flood_bangkok_spatial_cnn_infrastructure import augmented_arrays
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_spatial_cnn import SpatialCNN
from flood_bangkapi_surface_grid import TO_LONLAT, grid_spec
from flood_bangkapi_surface_model import OUTPUT, RAIN_FEATURES, rainfall_by_date, scenario_rain, write_csv


ARTIFACT = "surface_citywide_rain_scenario_stress_v1"
AS_OF_DATE = date(2025, 10, 1)
SCENARIOS = {
    "dry": (0, 0, 0, 0),
    "20mm_2h": (10, 2, 0, 0),
    "40mm_2h": (20, 2, 0, 0),
    "40mm_4h": (10, 4, 0, 0),
    "80mm_4h": (20, 4, 0, 0),
    "20mm_2h_wet_prior": (10, 2, 20, 40),
}
MODELS = ("canal_cnn", "cnn_gnn")


def weather(rate: float, duration: int, prev24: float, prev72: float) -> dict:
    result = scenario_rain(rate, duration)
    result["rain_prev24h_mm"] = prev24
    result["rain_prev72h_mm"] = prev72
    result["rain_today_prev24h_mm"] = result["rain_today_mm"] + prev24
    return result


def scene(raw: np.ndarray, values: dict) -> np.ndarray:
    copy = np.asarray(raw, dtype=np.float32).copy()
    mask = copy[0, -1] > 0
    for i, key in enumerate(RAIN_FEATURES):
        copy[0, 10 + i, mask] = math.log1p(values[key])
    return copy


def percentile(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 2:
        return np.ones_like(scores)
    return 1 - (rankdata(-scores, method="average") - 1) / (len(scores) - 1)


def training_support(codes: tuple[str, ...]) -> dict:
    rows = []
    for code in codes:
        rain = rainfall_by_date(date(2022, 1, 1), date(2025, 7, 31),
                                CACHE / f"rain_{code}_2021_2025.json")
        rows.extend([item[key] for key in RAIN_FEATURES] for item in rain.values())
    values = np.asarray(rows, dtype=np.float32)
    return {"training_district_days": len(values),
            "feature_quantiles": {
                key: {str(q): round(float(np.quantile(values[:, i], q)), 2)
                      for q in (.95, .99, 1.0)}
                for i, key in enumerate(RAIN_FEATURES)},
            "scenarios": {
                name: {"district_days_at_least_today_and_peak_hour": int(np.count_nonzero(
                    (values[:, 0] >= v["rain_today_mm"]) &
                    (values[:, 1] >= v["rain_peak_1h_mm"]))),
                       "district_days_at_least_all_rain_summaries": int(np.count_nonzero(
                           np.all(values >= np.asarray([v[k] for k in RAIN_FEATURES]), axis=1)))}
                for name, config in SCENARIOS.items()
                for v in [weather(*config)]}}


def main() -> None:
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    districts = {d["code"]: d for d in load_districts()}
    meta = prepare_cache(districts)
    arrays = augmented_arrays(meta)
    graphs = load_graphs(meta, device)
    canal_ck = torch.load(
        OUTPUT / "surface_crossdistrict_infrastructure_ablation_citywide_v1_canal_only.pt",
        map_location=device, weights_only=False)
    graph_ck = torch.load(OUTPUT / "surface_crossdistrict_cnn_gnn_flow_v1.pt",
                          map_location=device, weights_only=False)
    canal = SpatialCNN(CHANNELS + 8).to(device)
    canal.load_state_dict(canal_ck["state_dict"])
    canal.eval()
    hybrid = FlowHybrid(SpatialCNN(CHANNELS + 1)).to(device)
    hybrid.load_state_dict(graph_ck["state_dict"])
    hybrid.eval()
    canal_metrics = json.loads((OUTPUT / "surface_crossdistrict_infrastructure_ablation_citywide_v1_metrics.json").read_text())
    graph_metrics = json.loads((OUTPUT / "surface_crossdistrict_cnn_gnn_flow_v1_metrics.json").read_text())
    biases = {
        "canal_cnn": {r["code"]: r["log_odds_shift"] for r in canal_metrics["models"]["canal_only"]["evaluated_districts"]},
        "cnn_gnn": {r["code"]: r["log_odds_shift"] for r in graph_metrics["evaluated_districts"]},
    }
    cell_rows, district_rows = [], []
    for code, district in districts.items():
        x, _ = arrays[code]
        rr = np.asarray(meta[code]["valid_rows"])
        cc = np.asarray(meta[code]["valid_cols"])
        spec = grid_spec(district["polygon"])
        centre_x = spec["west"] + (cc + 0.5) * 250
        centre_y = spec["north"] - (rr + 0.5) * 250
        lon, lat = TO_LONLAT.transform(centre_x, centre_y)
        with np.load(CACHE / "infrastructure_static_250m_v1" / f"{code}.npz") as features:
            asset = features["values"]
        base_day = np.asarray(x[CAL_N:CAL_N + 1], dtype=np.float32)
        if AS_OF_DATE != date(2022, 1, 1).fromordinal(date(2022, 1, 1).toordinal() + CAL_N):
            raise ValueError("Scene date no longer matches cache index")
        scores = {model: {} for model in MODELS}
        for name, config in SCENARIOS.items():
            raw = scene(base_day, weather(*config))
            canal_x = torch.from_numpy(transform(raw, canal_ck["stats"],
                                                 tuple(canal_ck["masked_channels"]))).to(device)
            graph_x = torch.from_numpy(transform(raw, graph_ck["stats"])).to(device)
            with torch.no_grad():
                canal_logits = canal(canal_x).cpu().numpy()[0, 0, rr, cc]
                graph_logits = hybrid(graph_x, graphs[code])[0].cpu().numpy()[0, 0, rr, cc]
            for model, logits in (("canal_cnn", canal_logits), ("cnn_gnn", graph_logits)):
                scores[model][name] = expit(logits + biases[model][code])
        for model in MODELS:
            baseline = scores[model]["20mm_2h"]
            for name, probability in scores[model].items():
                pctl = percentile(probability)
                district_rows.append({"district_code": code, "district": district["name_en"],
                                      "model": model, "scenario": name, "cells": len(rr),
                                      "sum_proxy_score": round(float(probability.sum()), 5),
                                      "score_index_vs_20mm_2h": round(float(probability.sum() / baseline.sum()), 5),
                                      "rank_spearman_vs_20mm_2h": round(float(spearmanr(
                                          probability, baseline).statistic), 5),
                                      "fraction_cells_score_below_20mm_2h": round(float(
                                          np.mean(probability < baseline - 1e-9)), 5)})
                for i in range(len(rr)):
                    cell_rows.append({"district_code": code, "district": district["name_en"],
                                      "row": int(rr[i]), "col": int(cc[i]),
                                      "lon": round(float(lon[i]), 6), "lat": round(float(lat[i]), 6),
                                      "model": model, "scenario": name,
                                      "proxy_score": round(float(probability[i]), 7),
                                      "district_rank_percentile": round(float(pctl[i]), 5),
                                      "canal_distance_m": round(float(np.expm1(asset[i, 1])), 1),
                                      "pump_capacity_within_1km_reported_units": round(float(np.expm1(asset[i, 2])), 2),
                                      "underpass_within_250m": bool(asset[i, 5])})
        print(code, district["name_en"], flush=True)
    write_csv(OUTPUT / f"{ARTIFACT}_district_summary.csv", district_rows)
    write_csv(OUTPUT / f"{ARTIFACT}_cells.csv", cell_rows)
    payload = {"purpose": "inspect saved model response to synthetic daily rain summaries",
               "as_of_date": str(AS_OF_DATE),
               "scenarios": {name: weather(*config) for name, config in SCENARIOS.items()},
               "training_rain_support": training_support(tuple(code for code in districts if code not in HOLDOUT)),
               "models": {"canal_cnn": "saved canal-only 24-feature CNN",
                          "cnn_gnn": "saved frozen 17-feature CNN plus graph residual"},
               "score_meaning": "sum of locally intercept-adjusted complaint-proxy scores, not expected flooded cells",
               "limitations": ["Single daily rain summary per district; no intra-day water routing",
                               "No time-matched gate, pump, tide or pipe-flow state",
                               "Synthetic scenarios are not validated against actual flood extent",
                               "High-rain combinations may lie beyond the model training distribution",
                               "Local calibration underpredicts 2025 held-out complaint counts"]}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
