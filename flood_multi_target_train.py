"""Train shared spatial features with separate satellite, report and depth heads.

This is a training run, not a calibrated inundation or forward-time model.
Absent satellite polygons and absent reports are never used as dry labels.
"""

from __future__ import annotations

import json
import math
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flood_multi_target_data import AUDIT, PATCH_CODES
from flood_open_weak_data import FEATURE_NAMES, WEATHER_NAMES, OUTPUT, ROOT
from flood_open_weak_train import augment, prepare, weak_loss


DEST = OUTPUT / "multi_target"
SEED = 97
EPOCHS = 64
STEPS_PER_EPOCH = 48
TASK_PROBS = {
    "unosat2021": .10, "unosat2024": .025,
    "eos2025": .025, "eos2026": .05,
    "gistda2026": .25, "traffy": .45, "bma": .10,
}
URBAN_ONLY_PROBS = {"traffy": .82, "bma": .18}
HEAD_FOR = {"unosat2021": "unosat", "unosat2024": "unosat",
            "eos2025": "eos", "eos2026": "eos", "gistda2026": "gistda"}


class MultiTargetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(len(FEATURE_NAMES) + len(WEATHER_NAMES), 24, 3, padding=1),
            nn.ReLU(), nn.Conv2d(24, 16, 3, padding=1), nn.ReLU())
        self.satellite = nn.ModuleDict({name: nn.Conv2d(16, 1, 1)
                                        for name in ("unosat", "eos", "gistda")})
        self.urban = nn.Sequential(nn.Conv2d(20, 16, 3, padding=1), nn.ReLU())
        self.traffy = nn.Conv2d(16, 1, 1)
        self.bma = nn.Conv2d(16, 1, 1)
        self.depth = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.depth.weight)
        nn.init.zeros_(self.depth.bias)

    def forward(self, static: torch.Tensor, weather: torch.Tensor,
                history: torch.Tensor) -> dict[str, torch.Tensor]:
        if history.shape != (len(static), 2, *static.shape[-2:]):
            raise ValueError("History must contain score and availability maps")
        w = weather[:, :, None, None].expand(-1, -1, *static.shape[-2:])
        x = torch.cat((static[:, :7], w, static[:, 7:]), dim=1)
        latent = self.encoder(x)
        urban = self.urban(torch.cat((latent, static[:, 7:8], static[:, 5:6],
                                      history), dim=1))
        return {**{name: layer(latent)[:, 0] for name, layer in self.satellite.items()},
                "traffy": self.traffy(urban)[:, 0],
                "bma": self.bma(urban)[:, 0],
                "depth_cm": 15. + 5. * self.depth(urban)[:, 0]}


def normalize(raw_static: np.ndarray, raw_weather: np.ndarray, norm: dict):
    sm = np.asarray(norm["static_mean"], np.float32)[:, None, None]
    ss = np.asarray(norm["static_std"], np.float32)[:, None, None]
    wm = np.asarray(norm["weather_mean"], np.float32)
    ws = np.asarray(norm["weather_std"], np.float32)
    return (np.clip((raw_static-sm)/ss, -5, 5).astype(np.float32),
            np.clip((raw_weather-wm)/ws, -5, 5).astype(np.float32))


def load_urban(norm: dict, manifest: dict) -> tuple[dict, dict]:
    eos = {s["region"]: s for s in manifest["scenes"] if s["source"] == "eos2026"}
    audit = json.loads(AUDIT.read_text())
    result = {}
    counts = Counter()
    for region in PATCH_CODES:
        with np.load(ROOT / audit["patches"][region]["path"]) as z:
            reports, depths, mask, weather = (z[k] for k in
                ("reports", "depth_cm", "bma_mask", "weather"))
        with np.load(ROOT / eos[region]["path"]) as z:
            raw_static = z["static"].astype(np.float32)
        static, weather = normalize(raw_static, weather.astype(np.float32), norm)
        reports = reports.astype(np.uint8)
        depths = depths.astype(np.float32)
        mask = mask.astype(bool)
        if reports.shape != depths.shape or reports.shape[1:] != (80, 80):
            raise ValueError("Urban label shapes differ")
        if (np.any(reports[:, ~mask]) or np.isfinite(depths[:, ~mask]).any()
                or not np.all(np.isfinite(static)) or not np.all(np.isfinite(weather))):
            raise ValueError("Urban labels or features leave valid BMA cells")
        prefix = np.concatenate((np.zeros((1, 80, 80), np.uint16),
                                 np.cumsum(reports, axis=0, dtype=np.uint16)), axis=0)
        report_days = np.flatnonzero(reports.sum(axis=(1, 2)) > 0)
        bma_days = np.flatnonzero(np.isfinite(depths).any(axis=(1, 2)))
        result[region] = {"static": static, "weather": weather,
                          "reports": reports, "depths": depths, "mask": mask,
                          "history_prefix": prefix,
                          "report_days": report_days, "bma_days": bma_days}
        counts[f"{region}_report_days"] = len(report_days)
        counts[f"{region}_depth_days"] = len(bma_days)
        counts[f"{region}_report_cell_days"] = int(reports.sum())
        counts[f"{region}_depth_cell_days"] = int(np.isfinite(depths).sum())
    return result, dict(counts)


def urban_sample(item: dict, day: int, rng: np.random.Generator):
    prefix = item["history_prefix"]
    prior = prefix[day] - prefix[max(0, day-365)]
    history = np.stack((np.log1p(prior.astype(np.float32)) / math.log1p(20),
                        np.full((80, 80), min(day,365)/365, np.float32)))
    # Rotate all maps together, including masks, to preserve spatial alignment.
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    def move(x):
        x = np.rot90(x, k, axes=(-2, -1))
        return np.ascontiguousarray(np.flip(x, axis=-1) if flip else x)
    return (torch.from_numpy(move(item["static"])[None]),
            torch.from_numpy(item["weather"][day:day+1].copy()),
            torch.from_numpy(move(history)[None]),
            torch.from_numpy(move(item["reports"][day].astype(np.float32))),
            torch.from_numpy(move(item["depths"][day])),
            torch.from_numpy(move(item["mask"])))


def location_loss(logits: torch.Tensor, positives: torch.Tensor,
                  candidate: torch.Tensor) -> torch.Tensor:
    target = positives[candidate]
    if target.sum() <= 0:
        raise ValueError("Urban scene has no positive sites")
    return -(target / target.sum() * F.log_softmax(logits[candidate], dim=0)).sum() - math.log(int(candidate.sum()))


def train(mix: str = "all") -> dict:
    if mix not in ("all", "urban_only"):
        raise ValueError(f"Unknown task mix: {mix}")
    destination = DEST if mix == "all" else DEST / "urban_only"
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    satellite, norm = prepare()
    manifest = json.loads((OUTPUT / "data_audit.json").read_text())
    urban, urban_counts = load_urban(norm, manifest)
    groups = defaultdict(list)
    for item in satellite:
        groups[item["source"]].append(item)
    task_probs = TASK_PROBS if mix == "all" else URBAN_ONLY_PROBS
    steps_per_epoch = STEPS_PER_EPOCH if mix == "all" else 27
    if set(groups) != set(HEAD_FOR) or abs(sum(task_probs.values())-1) > 1e-8:
        raise ValueError("Task probabilities or satellite groups changed")
    bma_regions = [name for name, item in urban.items() if len(item["bma_days"])]
    if not bma_regions or any(len(item["report_days"]) == 0 for item in urban.values()):
        raise ValueError("Missing urban task examples")
    net = MultiTargetCNN()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=.0001)
    tasks = tuple(task_probs)
    probs = np.asarray(list(task_probs.values()), np.float64)
    draws = Counter()
    curves = []
    for epoch in range(1, EPOCHS+1):
        net.train()
        epoch_losses = defaultdict(list)
        for _ in range(steps_per_epoch):
            task = str(rng.choice(tasks, p=probs))
            optimizer.zero_grad()
            if task in HEAD_FOR:
                pool = groups[task]
                if task == "unosat2021":
                    dry_pool = [s for s in pool if s["wet"].sum() <= 1]
                    wet_pool = [s for s in pool if s["wet"].sum() > 1]
                    pool = dry_pool if dry_pool and rng.random() < .15 else wet_pool
                item = pool[int(rng.integers(len(pool)))]
                spatial, weather, wet, observed = augment(item, rng)
                empty_history = torch.zeros((1, 2, 80, 80), dtype=torch.float32)
                prediction = net(spatial, weather, empty_history)[HEAD_FOR[task]][0]
                loss, pieces = weak_loss(prediction, wet, observed, task)
                draws[f"scene:{item['source']}:{item['region']}:{item['day']}"] += 1
                for key, value in pieces.items():
                    epoch_losses[f"{task}_{key}"].append(value)
            else:
                region = (str(rng.choice(bma_regions)) if task == "bma"
                          else str(rng.choice(list(urban))))
                item = urban[region]
                days = item["bma_days"] if task == "bma" else item["report_days"]
                day = int(rng.choice(days))
                spatial, weather, history, reports, depths, mask = urban_sample(item, day, rng)
                prediction = net(spatial, weather, history)
                if task == "traffy":
                    loss = location_loss(prediction["traffy"][0], reports, mask)
                else:
                    positive = torch.isfinite(depths).float()
                    rank = location_loss(prediction["bma"][0], positive, mask)
                    valid = torch.isfinite(depths)
                    severity = F.smooth_l1_loss(
                        (prediction["depth_cm"][0][valid]-depths[valid])/10,
                        torch.zeros_like(depths[valid]), beta=.5)
                    loss = rank + .2*severity
                    epoch_losses["bma_rank"].append(float(rank.detach()))
                    epoch_losses["bma_depth_scaled"].append(float(severity.detach()))
                draws[f"urban:{task}:{region}:{day}"] += 1
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss for {task}")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            draws[task] += 1
            epoch_losses[task].append(float(loss.detach()))
        record = {"epoch": epoch,
                  "learning_rate": round(float(optimizer.param_groups[0]["lr"]), 7),
                  **{key+"_mean": round(float(np.mean(values)), 5)
                     for key, values in epoch_losses.items()}}
        curves.append(record)
        scheduler.step()
        if epoch == 1 or epoch % 8 == 0 or epoch == EPOCHS:
            print(record, flush=True)
    path = destination / ("multi_target_cnn_all_open_sources.pt" if mix == "all"
                          else "urban_only_cnn.pt")
    torch.save({"state_dict": net.state_dict(),
                "model": "MultiTargetCNN", "static_features": FEATURE_NAMES,
                "weather_features": WEATHER_NAMES, "normalization": norm,
                "source_heads": HEAD_FOR, "task_probabilities": task_probs,
                "urban_train_end": "2025-09-30",
                "label_semantics": "source-specific mapped-water and report-location rankings; depth conditional on matched BMA incident"}, path)
    report = {"trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "checkpoint": str(path.relative_to(ROOT)),
              "parameter_count": sum(p.numel() for p in net.parameters()),
              "steps": EPOCHS*steps_per_epoch,
              "training_mix": mix,
              "satellite_scene_counts": dict(Counter(s["source"] for s in satellite)),
              "urban_training_counts": urban_counts,
              "task_probabilities": task_probs,
              "task_draws": {key: draws[key] for key in tasks},
              "distinct_scene_draws": sum(1 for key in draws if key.startswith("scene:")),
              "distinct_urban_draws": sum(1 for key in draws if key.startswith("urban:")),
              "curve": curves,
              "limits": ["training diagnostics only, no held-out performance claim",
                         "full-day rain may occur after report/satellite pass",
                         "2026 satellite training precludes a forward-time 2025 test",
                         "BMA depth positions use a 2026 risk-point inventory"]}
    (destination / "training_run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", choices=("all", "urban_only"), default="all")
    result = train(parser.parse_args().mix)
    print({key: result[key] for key in ("checkpoint", "steps", "satellite_scene_counts",
                                        "urban_training_counts", "task_draws")})
