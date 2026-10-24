"""Retrospective Q4 2025 urban-label diagnostic for joint versus urban-only CNN.

The joint network saw 2026 satellite maps, and this Q4 period was inspected
in prior work. These are target-specific diagnostics, not a prospective test.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta, datetime, timezone

import numpy as np
import torch

from flood_bangkok_multitask_depth_cnn import CODES, matched_depths
from flood_bangkok_transfer import CACHE as BMA_CACHE, load_districts
from flood_bangkapi_surface_grid import GRID_M, TO_UTM
from flood_bangkapi_surface_model import RAIN_FEATURES, rainfall_by_date
from flood_multi_target_data import AUDIT, PATCH_CODES, START, END, complaint_grid
from flood_multi_target_train import DEST, MultiTargetCNN, normalize
from flood_open_weak_bangkok_holdout_eval import selected
from flood_open_weak_data import FEATURE_NAMES, OUTPUT, ROOT, all_specs


FIRST = END + timedelta(days=1)
LAST = date(2025, 12, 31)
Q4_DAYS = [FIRST + timedelta(days=i) for i in range((LAST-FIRST).days+1)]
OUT = DEST / "q4_2025_urban_diagnostic.json"
CHECKPOINTS = {
    "joint_all_targets": DEST / "multi_target_cnn_all_open_sources.pt",
    "urban_only": DEST / "urban_only" / "urban_only_cnn.pt",
}


def q4_weather(code: str) -> np.ndarray:
    path = BMA_CACHE / f"rain_{code}_2021_2025.json"
    raw = rainfall_by_date(FIRST, LAST, path)
    rows = []
    for day in Q4_DAYS:
        rows.append([math.log1p(max(0., raw[day][key])) for key in RAIN_FEATURES]
                    + [math.sin(2*math.pi*day.month/12),
                       math.cos(2*math.pi*day.month/12)])
    return np.asarray(rows, np.float32)


def bma_q4_sites(districts: list[dict], patch_specs: dict, masks: dict) -> dict:
    _, rows, audit = matched_depths({d["code"]: d for d in districts if d["code"] in CODES})
    grids = {name: np.full((len(Q4_DAYS), 80, 80), np.nan, np.float32)
             for name in PATCH_CODES}
    accepted = {name: 0 for name in PATCH_CODES}
    for row in rows:
        day = date.fromisoformat(row["date"])
        if day < FIRST or day > LAST:
            continue
        i = (day-FIRST).days
        x, y = TO_UTM.transform(row["inventory_lon"], row["inventory_lat"])
        for name, spec in patch_specs.items():
            if name not in grids:
                continue
            west, south, east, north = spec["utm_bounds"]
            if not (west <= x < east and south < y <= north):
                continue
            rr = math.floor((north-y)/GRID_M)
            cc = math.floor((x-west)/GRID_M)
            if not (0 <= rr < 80 and 0 <= cc < 80 and masks[name][rr,cc]):
                continue
            old = grids[name][i,rr,cc]
            grids[name][i,rr,cc] = max(float(row["depth_cm"]), old) if np.isfinite(old) else row["depth_cm"]
            accepted[name] += 1
    return {"grids": grids, "accepted_rows": accepted, "source_audit": audit}


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["model"] != "MultiTargetCNN" or tuple(checkpoint["static_features"]) != FEATURE_NAMES:
        raise ValueError("Checkpoint model or static features changed")
    model = MultiTargetCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["normalization"]


def capture(scores: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    if scores.ndim == 2:
        scores = np.broadcast_to(scores, truth.shape)
    if scores.shape != truth.shape:
        raise ValueError("Score and truth shape differ")
    positive = int(truth[:, mask].sum())
    if not positive:
        return {"positive_cell_days": 0, "top_10pct_capture": None}
    daily_hits = [float(np.dot(selected(s[mask], .1), y[mask]))
                  for s, y in zip(scores, truth)]
    daily_positives = truth[:, mask].sum(axis=1).astype(int).tolist()
    hits = sum(daily_hits)
    return {"positive_cell_days": positive,
            "captured_fractional_positives": round(hits, 5),
            "top_10pct_capture": round(hits/positive, 5),
            "daily_positives": daily_positives,
            "daily_captured": [round(v, 5) for v in daily_hits]}


def main() -> None:
    torch.set_num_threads(4)
    audit = json.loads(AUDIT.read_text())
    districts = load_districts()
    specs = all_specs()
    training_length = (END-START).days+1
    raw = {}
    masks = {}
    for name, code in PATCH_CODES.items():
        with np.load(ROOT / audit["patches"][name]["path"]) as z:
            train_reports = z["reports"].astype(np.uint8)
            mask = z["bma_mask"].astype(bool)
        full, _ = complaint_grid(name, specs[name], districts, 1461)
        full[:, ~mask] = 0
        if not np.array_equal(full[:training_length], train_reports):
            raise ValueError(f"Training report labels changed for {name}")
        prefix = np.concatenate((np.zeros((1,80,80), np.uint16),
                                 np.cumsum(full, axis=0, dtype=np.uint16)), axis=0)
        history = np.stack([prefix[i]-prefix[i-365] for i in range(training_length,1461)])
        manifest = json.loads((OUTPUT / "data_audit.json").read_text())
        eos = next(s for s in manifest["scenes"] if s["source"] == "eos2026" and s["region"] == name)
        with np.load(ROOT / eos["path"]) as z:
            static = z["static"].astype(np.float32)
        raw[name] = {"static": static, "weather": q4_weather(code),
                     "reports": full[training_length:], "history": history,
                     "mask": mask}
        masks[name] = mask
    bma = bma_q4_sites(districts, specs, masks)
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "period": [str(FIRST), str(LAST)],
              "interpretation": "retrospective cross-time diagnostic; 2026 satellite scenes were used in joint training; prior work inspected Q4 labels",
              "candidate_budget": "top 10% of BMA candidate cells in each patch on each day",
              "bma_matched_rows": bma["accepted_rows"],
              "models": {}, "baselines": {}}
    for name, item in raw.items():
        truth = item["reports"]
        mask = item["mask"]
        result["baselines"][name] = {
            "built_fraction": capture(item["static"][7], truth, mask),
            "crop_fraction": capture(item["static"][8], truth, mask),
            "rolling_report_history": capture(item["history"], truth, mask),
            "bma_site": {
                "built_fraction": capture(item["static"][7],
                    np.isfinite(bma["grids"][name]).astype(np.uint8), mask),
                "rolling_report_history": capture(item["history"],
                    np.isfinite(bma["grids"][name]).astype(np.uint8), mask)}}
    for model_name, path in CHECKPOINTS.items():
        model, norm = load_model(path)
        model_result = {"checkpoint": str(path.relative_to(ROOT)), "patches": {}}
        for name, item in raw.items():
            static, weather = normalize(item["static"], item["weather"], norm)
            scores, bma_scores, depths = [], [], []
            with torch.no_grad():
                for i in range(len(Q4_DAYS)):
                    prior = np.log1p(item["history"][i].astype(np.float32))/math.log1p(20)
                    history = np.stack((prior, np.ones((80,80), np.float32)))
                    output = model(torch.from_numpy(static[None]),
                                   torch.from_numpy(weather[i:i+1]),
                                   torch.from_numpy(history[None]))
                    scores.append(output["traffy"][0].numpy())
                    bma_scores.append(output["bma"][0].numpy())
                    depths.append(output["depth_cm"][0].numpy())
            scores, bma_scores, depths = np.stack(scores), np.stack(bma_scores), np.stack(depths)
            report_metric = capture(scores, item["reports"], item["mask"])
            known_depth = bma["grids"][name]
            bma_metric = capture(bma_scores, np.isfinite(known_depth).astype(np.uint8), item["mask"])
            valid = np.isfinite(known_depth)
            depth_metric = ({"matched_cell_days": int(valid.sum()),
                             "mae_cm": round(float(np.mean(abs(depths[valid]-known_depth[valid]))), 4),
                             "constant_15cm_mae_cm": round(float(np.mean(abs(15-known_depth[valid]))), 4)}
                            if valid.any() else {"matched_cell_days": 0})
            model_result["patches"][name] = {
                "traffy": report_metric, "bma_site": bma_metric,
                "bma_depth_conditional": depth_metric}
        result["models"][model_name] = model_result
    def pooled(section, key):
        rows = [section[name][key] for name in PATCH_CODES]
        total = sum(r["positive_cell_days"] for r in rows)
        daily_positives = np.stack([r["daily_positives"] for r in rows]).sum(axis=0)
        daily_captured = np.stack([r["daily_captured"] for r in rows]).sum(axis=0)
        return {"positive_cell_days": total,
                "top_10pct_capture": round(sum(r["captured_fractional_positives"] for r in rows)/total,5),
                "daily_positives": daily_positives.tolist(),
                "daily_captured": np.round(daily_captured,5).tolist()}
    result["pooled_report_capture"] = {
        **{name: pooled({p: v["patches"][p] for p in PATCH_CODES}, "traffy")
           for name, v in result["models"].items()},
        **{name: pooled(result["baselines"], name)
           for name in ("built_fraction", "crop_fraction", "rolling_report_history")}}
    joint = result["pooled_report_capture"]["joint_all_targets"]
    other = result["pooled_report_capture"]
    blocks = [np.arange(i, min(i+7, len(Q4_DAYS)))
              for i in range(0, len(Q4_DAYS), 7)]
    rng = np.random.default_rng(112)
    intervals = {}
    for comparator in ("urban_only", "rolling_report_history"):
        differences = []
        for _ in range(4000):
            chosen = rng.integers(len(blocks), size=len(blocks))
            indices = np.concatenate([blocks[i] for i in chosen])
            positives = np.asarray(joint["daily_positives"])[indices].sum()
            if positives <= 0:
                continue
            a = np.asarray(joint["daily_captured"])[indices].sum()
            b = np.asarray(other[comparator]["daily_captured"])[indices].sum()
            differences.append(float((a-b)/positives))
        intervals[comparator] = {
            "point_difference": round(joint["top_10pct_capture"]-
                                      other[comparator]["top_10pct_capture"],5),
            "weekly_block_bootstrap_95pct": [round(float(x),5) for x in
                np.quantile(differences,[.025,.975])],
            "blocks": len(blocks)}
    result["paired_capture_differences"] = intervals
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("pooled", result["pooled_report_capture"], flush=True)
    for name in CHECKPOINTS:
        print(name, result["models"][name]["patches"], flush=True)


if __name__ == "__main__":
    main()
