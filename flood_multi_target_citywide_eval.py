"""Q4 2025 Bangkok-wide report-ranking diagnostic for two frozen CNNs.

The Q4 labels were inspected in earlier work and are positive reports, not
verified wet/dry observations. This is a retrospective geography diagnostic.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import torch
from shapely.geometry import box
from shapely.ops import unary_union

from flood_bangkok_transfer import load_districts
from flood_open_weak_bangkok_holdout_eval import bma_fraction, selected
from flood_open_weak_data import ROOT, all_specs, load_static
from flood_multi_target_citywide_data import OUT, specs_and_masks
from flood_multi_target_data import END, START, complaint_grid
from flood_multi_target_q4_diagnostic import (
    CHECKPOINTS, FIRST, Q4_DAYS, load_model, q4_weather,
)
from flood_multi_target_train import normalize


OUT_JSON = OUT / "q4_2025_citywide_report_ranking.json"
N_DAYS = (END - START).days + 1 + len(Q4_DAYS)


def district_assignment(spec: dict, districts: list[dict], candidate: np.ndarray) -> np.ndarray:
    """Give every BMA edge cell to the district with greatest cell overlap."""
    identity = np.full((80, 80), -1, np.int16)
    greatest = np.zeros((80, 80), np.float32)
    tile = box(*spec["utm_bounds"])
    for i, district in enumerate(districts):
        if not district["polygon"].intersects(tile):
            continue
        fraction = bma_fraction(spec, district["polygon"]).astype(np.float32)
        better = fraction > greatest
        identity[better] = i
        greatest[better] = fraction[better]
    if np.any(candidate & (identity < 0)):
        raise ValueError("A Bangkok candidate cell has no district overlap")
    return identity


def build_arrays():
    specs, areas = specs_and_masks()
    districts = load_districts()
    bma = unary_union([d["polygon"] for d in districts])
    original = all_specs()
    trained = unary_union([box(*original[name]["utm_bounds"])
                           for name in ("bang_kapi_core", "nong_chok")])
    # Two encoder convolutions plus the urban convolution see a 3-cell radius.
    excluded = trained.buffer(750)
    training_length = (END - START).days + 1
    records, audit = [], {}
    for name, spec in specs.items():
        candidate = bma_fraction(spec, bma) > .02
        district_id = district_assignment(spec, districts, candidate)
        new = candidate & (bma_fraction(spec, excluded) == 0)
        reports, contributors = complaint_grid(name, spec, districts, N_DAYS)
        reports[:, ~candidate] = 0
        prefix = np.concatenate((np.zeros((1, 80, 80), np.uint16),
                                 np.cumsum(reports, axis=0, dtype=np.uint16)), axis=0)
        history = np.stack([prefix[i] - prefix[i-365]
                            for i in range(training_length, N_DAYS)]).astype(np.uint16)
        if (not np.array_equal(history[0], reports[training_length-365:training_length].sum(axis=0))
            or not np.array_equal(history[-1], reports[N_DAYS-366:N_DAYS-1].sum(axis=0))):
            raise ValueError(f"Rolling history includes or omits a scored day: {name}")
        truth = reports[training_length:]
        # Open-Meteo is a gridded archive sampled at district representative
        # points. Use the nearest such point to the Bangkok portion of a tile.
        point = bma.intersection(box(*spec["utm_bounds"])).centroid
        rain_district = min(districts, key=lambda d:
                            d["polygon"].representative_point().distance(point))
        weather = q4_weather(rain_district["code"])
        static = load_static(name)
        if (static.shape != (13, 80, 80) or
            truth.shape != (len(Q4_DAYS), 80, 80) or
            not np.isfinite(static).all() or not np.isfinite(weather).all() or
            np.any(truth[:, ~candidate])):
            raise ValueError(f"Invalid tile features or labels: {name}")
        records.append({"name": name, "static": static, "weather": weather,
                        "history": history, "truth": truth,
                        "candidate": candidate, "new": new, "district": district_id})
        audit[name] = {"candidate_cells": int(candidate.sum()),
                       "new_geography_cells": int(new.sum()),
                       "q4_report_positive_cell_days": int(truth.sum()),
                       "rain_proxy_district": rain_district["name_en"],
                       "rain_proxy_code": rain_district["code"],
                       "source_districts": [d["district_code"] for d in contributors],
                       **areas[name]}
        print("labels", name, audit[name]["candidate_cells"],
              audit[name]["q4_report_positive_cell_days"], flush=True)
    return records, districts, audit


def score_models(records: list[dict]) -> dict[str, np.ndarray]:
    torch.set_num_threads(4)
    outputs = {}
    for model_name, path in CHECKPOINTS.items():
        model, norm = load_model(path)
        per_tile = []
        with torch.no_grad():
            for item in records:
                static, weather = normalize(item["static"], item["weather"], norm)
                daily = []
                for start in range(0, len(Q4_DAYS), 8):
                    stop = min(start + 8, len(Q4_DAYS))
                    prior = (np.log1p(item["history"][start:stop].astype(np.float32)) /
                             math.log1p(20))
                    history = np.stack((prior, np.ones_like(prior)), axis=1)
                    batch_static = np.broadcast_to(static, (stop-start, *static.shape)).copy()
                    result = model(torch.from_numpy(batch_static),
                                   torch.from_numpy(weather[start:stop]),
                                   torch.from_numpy(history))
                    daily.append(result["traffy"].numpy())
                tile_score = np.concatenate(daily)
                if not np.isfinite(tile_score).all():
                    raise ValueError(f"Nonfinite model output in {item['name']}")
                per_tile.append(tile_score.reshape(len(Q4_DAYS), -1))
                print("scored", model_name, item["name"], flush=True)
        outputs[model_name] = np.concatenate(per_tile, axis=1)
    return outputs


def measure(scores: np.ndarray, truth: np.ndarray, select: np.ndarray) -> dict:
    if scores.ndim == 1:
        scores = np.broadcast_to(scores, truth.shape)
    indices = np.flatnonzero(select)
    positives = truth[:, indices].sum(axis=1, dtype=np.int64)
    if not len(indices):
        return {"candidate_cells": 0, "positive_cell_days": 0,
                "top_10pct_capture": None}
    hits = np.empty(len(Q4_DAYS), np.float64)
    for i in range(len(Q4_DAYS)):
        hits[i] = np.dot(selected(scores[i, indices], .1), truth[i, indices])
    total = int(positives.sum())
    return {"candidate_cells": int(len(indices)),
            "positive_cell_days": total,
            "positive_days": int((positives > 0).sum()),
            "captured_fractional_positives": round(float(hits.sum()), 5),
            "top_10pct_capture": round(float(hits.sum()/total), 5) if total else None,
            "daily_positive": positives.tolist(),
            "daily_captured": np.round(hits, 5).tolist()}


def paired_interval(a: dict, b: dict, seed: int = 112) -> list[float] | None:
    if not a["positive_cell_days"]:
        return None
    blocks = [np.arange(i, min(i+7, len(Q4_DAYS)))
              for i in range(0, len(Q4_DAYS), 7)]
    rng = np.random.default_rng(seed)
    positives = np.asarray(a["daily_positive"])
    first = np.asarray(a["daily_captured"])
    second = np.asarray(b["daily_captured"])
    differences = []
    for _ in range(2000):
        indices = np.concatenate([blocks[k] for k in
                                  rng.integers(len(blocks), size=len(blocks))])
        denom = positives[indices].sum()
        if denom:
            differences.append((first[indices].sum()-second[indices].sum())/denom)
    return [round(float(v), 5) for v in np.quantile(differences, [.025, .975])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, districts, data_audit = build_arrays()
    model_scores = score_models(records)
    n_days = len(Q4_DAYS)
    candidate = np.concatenate([r["candidate"].ravel() for r in records])
    new = np.concatenate([r["new"].ravel() for r in records])
    district_id = np.concatenate([r["district"].ravel() for r in records])
    truth = np.concatenate([r["truth"].reshape(n_days, -1) for r in records], axis=1)
    history = np.concatenate([r["history"].reshape(n_days, -1) for r in records], axis=1)
    built = np.concatenate([r["static"][7].ravel() for r in records])
    scores = {**model_scores,
              "rolling_report_history": history,
              "built_fraction": built}
    if sum((candidate & (district_id == i)).any() for i in range(len(districts))) != 50:
        raise ValueError("Fewer than 50 districts have candidates")
    if int(truth[:, candidate].sum()) != sum(x["q4_report_positive_cell_days"]
                                            for x in data_audit.values()):
        raise ValueError("Citywide target sum changed")
    scopes = {"all_bangkok_citywide_queue": candidate,
              "outside_training_patches_750m_buffer_citywide_queue": new}
    scopes.update({f"district_{d['code']}": candidate & (district_id == i)
                   for i, d in enumerate(districts)})
    scopes.update({f"new_geography_district_{d['code']}": new & (district_id == i)
                   for i, d in enumerate(districts)})
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "period": [str(Q4_DAYS[0]), str(Q4_DAYS[-1])],
              "metric": "share of Traffy report-positive 250 m cell-days captured by daily top 10% queue within each scope; score ties split fractionally",
              "interpretation": "retrospective diagnostic; Q4 labels previously inspected; absent complaint does not establish dry; 2026 satellite labels and current OSM can postdate Q4; daily rain can include hours after complaint",
              "rain_source": "Open-Meteo archive sampled at one district representative point per tile, selected by proximity to Bangkok portion of tile",
              "training_patch_exclusion": "750 m buffer around Bang Kapi and Nong Chok 20 km training patches, matching the urban CNN's three 3x3 convolution receptive-field radius; excludes any 250 m cell touching buffered patch at 50 m raster precision",
              "tile_audit": data_audit,
              "district_names": {d["code"]: d["name_en"] for d in districts},
              "results": {}}
    for scope, mask in scopes.items():
        row = {name: measure(score, truth, mask) for name, score in scores.items()}
        row["joint_minus_urban_only_95pct_weekly_block_interval"] = paired_interval(
            row["joint_all_targets"], row["urban_only"])
        row["joint_minus_history_95pct_weekly_block_interval"] = paired_interval(
            row["joint_all_targets"], row["rolling_report_history"])
        result["results"][scope] = row
        print("metric", scope, row["joint_all_targets"]["positive_cell_days"],
              {name: value["top_10pct_capture"] for name, value in row.items()
               if isinstance(value, dict)}, flush=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print("saved", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
