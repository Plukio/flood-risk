"""Matched six-hour reported-site ranking with and without pre-cutoff rain.

This experiment retains the 250 m physical-site outcome and geographic folds.
It uses only rain observations timestamped before each six-hour decision time.
Report timestamps are treated as Bangkok local time; that source convention
has not been independently verified. Unreported sites are unlabeled rivals.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F

from flood_citywide_history_cnn_transfer import RADIUS, patches
from flood_citywide_history_ranker import N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid
from flood_citywide_physical_site_data import build
from flood_citywide_spatial_novel_queue import (SpatialNovelCNN, day_maps,
                                                 queue_weights, static_image)
from flood_multi_target_citywide_data import OUT
from flood_open_weak_data import ROOT


RESULT = OUT / "q4_2025_physical_site_hourly_rain_fivefold.json"
WINDOWS = (1, 3, 6, 24, 72)
ALPHA = .5  # Fixed before scoring, identical in both learned arms.
EPOCHS = 2
SEED = 30841
HOURS_PER_SLOT = 6
RAINFALL_START = datetime(2021, 1, 1)


class RainSiteCNN(SpatialNovelCNN):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(16 + len(WINDOWS), 8, 1)


def matched_models(seed: int):
    torch.manual_seed(seed)
    plain = SpatialNovelCNN()
    rain = RainSiteCNN()
    with torch.no_grad():
        rain.entry.weight.zero_()
        rain.entry.weight[:, :14] = plain.entry.weight[:, :14]
        rain.entry.weight[:, 14+len(WINDOWS):] = plain.entry.weight[:, 14:]
        rain.entry.bias.copy_(plain.entry.bias)
        rain.steps.load_state_dict(plain.steps.state_dict())
        rain.head.load_state_dict(plain.head.state_dict())
        for model in (plain, rain):
            model.head.weight.zero_()
            model.head.bias.zero_()
    return plain, rain


def rain_archive(districts: list[dict]):
    series = []
    times = None
    for district in districts:
        path = ROOT / "data/flood_bangkok_transfer" / f"rain_{district['code']}_2021_2025.json"
        record = json.loads(path.read_text())
        if record.get("timezone") != "Asia/Bangkok":
            raise ValueError(f"Rain archive timezone: {path}")
        hourly = record["hourly"]
        if times is None:
            times = hourly["time"]
        elif times != hourly["time"]:
            raise ValueError("Rain archives have different hour indexes")
        values = np.asarray(hourly["precipitation"], np.float32)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Missing or invalid rain: {path}")
        series.append(values)
    if times[0] != "2021-01-01T00:00" or times[-1] != "2025-12-31T23:00":
        raise ValueError("Unexpected rain date coverage")
    rainfall = np.stack(series, axis=1)
    return np.concatenate((np.zeros((1, len(districts)), np.float64),
                           np.cumsum(rainfall, axis=0, dtype=np.float64)))


def slot_hour(event: dict) -> int:
    stamp = datetime.fromisoformat(event["first_report_time"])
    if stamp.tzinfo is not None:
        raise ValueError("Traffy timestamp convention changed")
    cutoff = stamp.replace(hour=(stamp.hour // HOURS_PER_SLOT)*HOURS_PER_SLOT,
                           minute=0, second=0, microsecond=0)
    hour = int((cutoff - RAINFALL_START).total_seconds() // 3600)
    if hour < max(WINDOWS):
        raise ValueError("Insufficient pre-cutoff rain history")
    return hour


def rain_values(prefix: np.ndarray, hour: int):
    # A record at the cutoff hour is withheld. Only preceding archive hours enter.
    return np.stack([prefix[hour] - prefix[hour-w] for w in WINDOWS], axis=1).astype(np.float32)


def rain_image(prefix: np.ndarray, hour: int, district_grid: np.ndarray):
    by_district = np.log1p(rain_values(prefix, hour))
    image = np.zeros((len(WINDOWS), *SHAPE), np.float32)
    valid = district_grid >= 0
    image[:, valid] = by_district[district_grid[valid]].T
    return image


def with_rain(plain: np.ndarray | torch.Tensor,
              rain: np.ndarray | torch.Tensor):
    if isinstance(plain, torch.Tensor):
        return torch.cat((plain[:, :14], rain, plain[:, 14:]), dim=1)
    return np.concatenate((plain[:14], rain, plain[14:]), axis=0)


def grouped(events: list[dict], begin: int, end: int, allowed: np.ndarray):
    rows = defaultdict(list)
    for event in events:
        if (begin <= event["day_index"] < end and
                allowed[event["district_index"]] and event["new_site"]):
            rows[(event["day_index"], slot_hour(event))].append(event)
    return rows


def train(data: dict, events: list[dict], source: np.ndarray,
          static: np.ndarray, district: np.ndarray, rain_prefix: np.ndarray,
          district_grid: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    plain, rainy = matched_models(seed)
    models = (plain, rainy)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.02)
                  for m in models]
    flat = data["candidate_flat"]
    lookup = np.full(np.prod(SHAPE), -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    source_allowed = np.zeros(len(data["districts"]), bool)
    source_allowed[np.unique(district[source])] = True
    slots = grouped(events, TRAIN_START, VALID_START, source_allowed)
    losses = {"no_rain": [], "rain": []}
    n_pairs = 0
    for epoch in range(EPOCHS):
        running = [[], []]
        order = rng.permutation(len(slots))
        slot_keys = list(slots)
        for at in order:
            day, hour = slot_keys[at]
            image, _, simple, novel = day_maps(data, day, static)
            positive = np.unique([lookup[e["flat"]] for e in slots[(day, hour)]])
            positive = positive[(positive >= 0) & novel[positive]]
            if len(positive) == 0:
                continue
            positive = rng.choice(positive, min(16, len(positive)), replace=False)
            # Same-day reported cells cannot become unlabeled rivals in training.
            reported = data["reports"][day].ravel()[flat].astype(bool)
            pos, cmp = [], []
            for cell in positive:
                pool = np.flatnonzero(source & novel & ~reported &
                                      (district == district[cell]))
                if len(pool) < 2:
                    continue
                near = pool[np.argsort(np.abs(simple[pool]-simple[cell]))[:min(32, len(pool))]]
                hard = pool[np.argsort(simple[pool])[-max(2, len(pool)//5):]]
                pos.extend((cell, cell))
                cmp.extend((rng.choice(near), rng.choice(hard)))
            if not pos:
                continue
            pos = np.asarray(pos, np.int32)
            cmp = np.asarray(cmp, np.int32)
            cells = np.r_[pos, cmp]
            plain_batch = patches(image, flat, cells)
            if torch.any(plain_batch[:len(pos), :4, RADIUS, RADIUS] != 0):
                raise ValueError("Own-cell pre-day report history leaked")
            rain_layer = rain_image(rain_prefix, hour, district_grid)
            rain_batch = patches(rain_layer, flat, cells)
            fixed = torch.from_numpy(simple[cells].copy())
            for index, (model, opt, batch) in enumerate(zip(
                    models, optimizers,
                    (plain_batch, with_rain(plain_batch, rain_batch)))):
                model.train()
                residual = model.patch_scores(batch)
                score = fixed + residual
                n = len(pos)
                loss = F.softplus(score[n:] - score[:n]).mean()
                loss = loss + .0005*residual.square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                running[index].append(float(loss.detach()))
            n_pairs += len(pos)
        for name, row in zip(losses, running):
            losses[name].append(round(float(np.mean(row)), 5))
        print("fold training epoch", epoch+1, losses, "pairs", n_pairs, flush=True)
    for m in models:
        m.eval()
    return models, {"slots": len(slots), "pairs_across_epochs": n_pairs,
                    "loss": losses}


def percent_rank(values: np.ndarray):
    return rankdata(values, method="average").astype(np.float32) / max(1, len(values))


def evaluate(data: dict, events: list[dict], allowed: np.ndarray,
             static: np.ndarray, models: tuple, rain_prefix: np.ndarray,
             district_grid: np.ndarray, begin: int, end: int):
    slots = grouped(events, begin, end, allowed)
    flat = data["candidate_flat"]
    lookup = np.full(np.prod(SHAPE), -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    candidate_district = data["district_id"].ravel()[flat]
    by_district = {i: np.flatnonzero(candidate_district == i)
                   for i in np.flatnonzero(allowed)}
    local = {i: {int(cell): at for at, cell in enumerate(cells)}
             for i, cells in by_district.items()}
    totals = {name: [0, 0.] for name in ("history", "no_rain", "rain")}
    wet = {name: [0, 0.] for name in totals}
    rain_by_event = []
    previous_day = None
    for day, hour in sorted(slots):
        if day != previous_day:
            image, base, simple, novel = day_maps(data, day, static)
            previous_day = day
        rain_layer = rain_image(rain_prefix, hour, district_grid)
        scores = [m.full_scores(im, data["candidate"])
                  for m, im in zip(models, (image, with_rain(image, rain_layer)))]
        district_events = defaultdict(list)
        for event in slots[(day, hour)]:
            district_events[event["district_index"]].append(event)
        six_hour = rain_values(rain_prefix, hour)[:, WINDOWS.index(6)]
        for district_id, these in district_events.items():
            ids = by_district[district_id]
            n = novel[ids]
            b, s = base[ids], simple[ids]
            weights = {"history": queue_weights(b, s, n, .2)}
            for name, score in zip(("no_rain", "rain"), scores):
                rank_simple = np.zeros(len(ids), np.float32)
                rank_model = np.zeros(len(ids), np.float32)
                rank_simple[n] = percent_rank(s[n])
                rank_model[n] = percent_rank((simple + score)[ids][n])
                mixed = (1-ALPHA)*rank_simple + ALPHA*rank_model
                weights[name] = queue_weights(b, mixed, n, .2)
            for event in these:
                where = local[district_id][lookup[event["flat"]]]
                mm = float(six_hour[district_id])
                rain_by_event.append(mm)
                for name, chosen in weights.items():
                    totals[name][0] += 1
                    totals[name][1] += float(chosen[where])
                    if mm >= 5:
                        wet[name][0] += 1
                        wet[name][1] += float(chosen[where])
    def present(rows):
        return {name: {"events": n, "hits": round(h, 3),
                       "capture_pct": round(100*h/n, 3) if n else None}
                for name, (n, h) in rows.items()}
    return {"all_new_sites": present(totals),
            "sites_after_at_least_5mm_previous_6h": present(wet),
            "event_rain_previous_6h_mm": {"median": round(float(np.median(rain_by_event)), 2)
                                          if rain_by_event else None,
                                          "fraction_at_least_5mm": round(float(np.mean(np.asarray(rain_by_event)>=5)), 3)
                                          if rain_by_event else None}}


def main():
    global HOURS_PER_SLOT, RESULT
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot-hours", type=int, choices=(1, 3, 6), default=6)
    args = parser.parse_args()
    HOURS_PER_SLOT = args.slot_hours
    if HOURS_PER_SLOT != 6:
        RESULT = OUT / f"q4_2025_physical_site_{HOURS_PER_SLOT}h_rain_fivefold.json"
    torch.set_num_threads(4)
    data = load_grid()
    events = build(data["candidate"], data["district_id"])[250]
    rain_prefix = rain_archive(data["districts"])
    district_grid = data["district_id"]
    flat = data["candidate_flat"]
    candidate_district = district_grid.ravel()[flat]
    folds = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    results = []
    for fold, members in enumerate(folds, 1):
        source = ~np.isin(candidate_district, members)
        static, _ = static_image(data, source)
        models, audit = train(data, events, source, static, candidate_district,
                              rain_prefix, district_grid, SEED+1000*fold)
        target_allowed = np.zeros(len(data["districts"]), bool)
        target_allowed[members] = True
        test = evaluate(data, events, target_allowed, static, models,
                        rain_prefix, district_grid, TEST_START, N_DAYS)
        results.append({"fold": fold,
                        "held_out_districts": [data["districts"][i]["code"] for i in members],
                        "training": audit, "q4": test})
        print("fold", fold, "Q4", test["all_new_sites"], flush=True)
    pooled = {}
    for group in ("all_new_sites", "sites_after_at_least_5mm_previous_6h"):
        pooled[group] = {}
        for arm in ("history", "no_rain", "rain"):
            n = sum(f["q4"][group][arm]["events"] for f in results)
            h = sum(f["q4"][group][arm]["hits"] for f in results)
            pooled[group][arm] = {"events": n, "hits": round(h, 3),
                                   "capture_pct": round(100*h/n, 3) if n else None}
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "target": f"First-in-year 250m Traffy site, first report in next {HOURS_PER_SLOT} hours",
              "rain_source": "Cached Open-Meteo hourly precipitation, 50 district-point proxies, Asia/Bangkok",
              "rain_windows_hours_before_cutoff": WINDOWS,
              "rain_cutoff": f"Strictly earlier archive hours than each {HOURS_PER_SLOT}-hour decision time",
              "report_time_assumption": "Naive Traffy timestamps treated as Bangkok local; independently unverified",
              "training": "2023-24 source districts; same six-hour examples and pairs for both CNNs",
              "test": f"Q4 2025 excluded districts, 10% of cells per district per {HOURS_PER_SLOT}-hour slot, 20% reserved for zero-own-history cells",
              "blend_alpha": ALPHA,
              "notes": ["Report sites are not verified flood extent",
                        "Unreported cells are unknown, not observed dry",
                        "Open-Meteo historical rain is reanalysis, not an archived live forecast",
                        "The subdaily queue is different from the published daily 10.01% task"],
              "folds": results, "pooled_q4": pooled}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print("pooled", json.dumps(pooled, indent=2), flush=True)


if __name__ == "__main__":
    main()
