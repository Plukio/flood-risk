"""Train a rain-conditioned spatial model on open weak labels.

The default fits every scene; --holdout-bangkok excludes both Bangkok patches
before fitting and normalization. Evaluation lives in a separate script.
"""

from __future__ import annotations

import json
import math
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flood_open_weak_data import CACHE, OUTPUT, ROOT, FEATURE_NAMES, WEATHER_NAMES
from flood_label_audit import scene_path, validate_scene


SEED = 83
EPOCHS = 48
STEPS_PER_EPOCH = 32
SOURCE_PROBABILITIES = {
    "unosat2021": .3,
    "unosat2024": .15,
    "eos2025": .15,
    "eos2026": .2,
    "gistda2026": .2,
}


class ConvBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1), nn.ReLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class WeatherUNet(nn.Module):
    """Weather enters at the bottleneck, preserving spatial feature channels."""

    def __init__(self, spatial_channels: int, weather_channels: int):
        super().__init__()
        self.encoder1 = ConvBlock(spatial_channels, 16)
        self.encoder2 = ConvBlock(16, 24)
        self.bottleneck = ConvBlock(24, 32)
        self.condition = nn.Sequential(nn.Linear(weather_channels, 32), nn.ReLU(),
                                       nn.Linear(32, 64))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        self.decoder2 = ConvBlock(32 + 24, 24)
        self.decoder1 = ConvBlock(24 + 16, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, spatial: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(spatial)
        e2 = self.encoder2(F.avg_pool2d(e1, 2))
        b = self.bottleneck(F.avg_pool2d(e2, 2))
        gamma, beta = self.condition(weather).chunk(2, dim=1)
        b = b * (1 + gamma.tanh()[:, :, None, None]) + beta[:, :, None, None]
        d2 = self.decoder2(torch.cat((F.interpolate(
            b, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2), dim=1))
        d1 = self.decoder1(torch.cat((F.interpolate(
            d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), dim=1))
        return self.output(d1)[:, 0]


class WeatherCNN(nn.Module):
    """Earlier three-layer spatial CNN with weather broadcast as input maps."""

    def __init__(self, spatial_channels: int, weather_channels: int):
        super().__init__()
        self.static_split = 7
        self.layers = nn.Sequential(
            nn.Conv2d(spatial_channels + weather_channels, 24, 3, padding=1),
            nn.ReLU(), nn.Conv2d(24, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1))

    def forward(self, spatial: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        width = spatial.shape[-2:]
        w = weather[:, :, None, None].expand(-1, -1, *width)
        x = torch.cat((spatial[:, :self.static_split], w,
                       spatial[:, self.static_split:]), dim=1)
        return self.layers(x)[:, 0]


def prepare(exclude_regions: tuple[str, ...] = ()) -> tuple[list[dict], dict]:
    manifest = json.loads((OUTPUT / "data_audit.json").read_text())
    if tuple(manifest["features"]) != FEATURE_NAMES or tuple(manifest["weather"]) != WEATHER_NAMES:
        raise ValueError("Training feature order differs from the data manifest")
    scenes = []
    for entry in manifest["scenes"]:
        if entry["region"] in exclude_regions:
            continue
        with np.load(scene_path(entry), allow_pickle=False) as z:
            data = {key: z[key].astype(np.float32) for key in
                    ("static", "weather", "wet", "observed")}
        validate_scene(entry, data, manifest)
        scenes.append({**entry, **data})
    by_source = Counter(s["source"] for s in scenes)
    expected = set(SOURCE_PROBABILITIES)
    if exclude_regions:
        expected.discard("eos2026")
    if set(by_source) != expected:
        raise ValueError(f"Expected sources {expected}, found {dict(by_source)}")
    # Each region's static grid contributes once to spatial normalization.
    unique = {s["region"]: s["static"] for s in scenes}
    static = np.stack(list(unique.values()))
    weather = np.stack([s["weather"] for s in scenes])
    sx_mean = static.mean(axis=(0, 2, 3)).astype(np.float32)
    sx_std = np.maximum(static.std(axis=(0, 2, 3)), .02).astype(np.float32)
    wx_mean = weather.mean(axis=0).astype(np.float32)
    wx_std = np.maximum(weather.std(axis=0), .05).astype(np.float32)
    for scene in scenes:
        scene["static"] = np.clip((scene["static"] - sx_mean[:, None, None]) /
                                  sx_std[:, None, None], -5, 5)
        scene["weather"] = np.clip((scene["weather"] - wx_mean) / wx_std, -5, 5)
    norm = {"static_mean": sx_mean.tolist(), "static_std": sx_std.tolist(),
            "weather_mean": wx_mean.tolist(), "weather_std": wx_std.tolist(),
            "distinct_regions": len(unique)}
    return scenes, norm


def augment(scene: dict, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    def move(a: np.ndarray) -> np.ndarray:
        a = np.rot90(a, k, axes=(-2, -1))
        if flip:
            a = np.flip(a, axis=-1)
        return np.ascontiguousarray(a)
    return (torch.from_numpy(move(scene["static"])[None]),
            torch.from_numpy(scene["weather"][None].copy()),
            torch.from_numpy(move(scene["wet"])),
            torch.from_numpy(move(scene["observed"])))


def weak_loss(logits: torch.Tensor, wet: torch.Tensor,
              observed: torch.Tensor, source: str) -> tuple[torch.Tensor, dict]:
    wet_mass = wet.sum()
    dry = (observed - wet).clamp_min(0)
    dry_mass = dry.sum()
    density = logits.new_zeros(())
    positive = logits.new_zeros(())
    negative = logits.new_zeros(())
    if wet_mass > 1:
        # This is a relative ranking of mapped positive area. It does not
        # label the remaining EOS/GISTDA cells dry.
        land = torch.ones_like(logits, dtype=torch.bool)
        log_prob = F.log_softmax(logits[land], dim=0)
        density = -(wet[land] / wet_mass * log_prob).sum() - math.log(land.sum().item())
        positive = (wet * F.softplus(-logits)).sum() / wet_mass
    if source.startswith("unosat") and dry_mass > 1:
        negative = (dry * F.softplus(logits)).sum() / dry_mass
    if source.startswith("unosat"):
        loss = density + .3 * positive + .1 * negative
    else:
        loss = density  # positive-only map; no invented dry-cell BCE term
    if wet_mass <= 1 and dry_mass <= 1:
        raise ValueError("Scene has no usable labels")
    return loss, {"density": float(density.detach()),
                  "positive": float(positive.detach()),
                  "weak_dry": float(negative.detach())}


def train(exclude_regions: tuple[str, ...] = (), output_dir: Path = OUTPUT,
          architecture: str = "unet") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    scenes, norm = prepare(exclude_regions)
    groups = defaultdict(list)
    for scene in scenes:
        groups[scene["source"]].append(scene)
    sources = tuple(s for s in SOURCE_PROBABILITIES if s in groups)
    probs = np.asarray([SOURCE_PROBABILITIES[s] for s in sources])
    probs /= probs.sum()
    model_class = {"unet": WeatherUNet, "cnn": WeatherCNN}.get(architecture)
    if model_class is None:
        raise ValueError(f"Unsupported model: {architecture}")
    model = model_class(len(FEATURE_NAMES), len(WEATHER_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=.0001)
    curve = []
    sampled = Counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        parts = defaultdict(list)
        for _ in range(STEPS_PER_EPOCH):
            source = str(rng.choice(sources, p=probs))
            pool = groups[source]
            if source == "unosat2021":
                positive_pool = [s for s in pool if s["wet"].sum() > 1]
                dry_pool = [s for s in pool if s["wet"].sum() <= 1]
                pool = dry_pool if dry_pool and rng.random() < .15 else positive_pool
            scene = pool[int(rng.integers(len(pool)))]
            spatial, weather, wet, observed = augment(scene, rng)
            optimizer.zero_grad()
            logits = model(spatial, weather)[0]
            loss, pieces = weak_loss(logits, wet, observed, source)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss for {scene['path']}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            sampled[source] += 1
            losses.append(float(loss.detach()))
            for key, value in pieces.items():
                parts[key].append(value)
        record = {"epoch": epoch, "learning_rate": round(optimizer.param_groups[0]["lr"], 7),
                  "train_loss_mean": round(float(np.mean(losses)), 5),
                  **{key + "_mean": round(float(np.mean(v)), 5)
                     for key, v in parts.items()}}
        curve.append(record)
        scheduler.step()
        if epoch == 1 or epoch % 4 == 0 or epoch == EPOCHS:
            print(record, flush=True)
    checkpoint = output_dir / (f"weather_{architecture}_bangkok_holdout.pt" if exclude_regions
                               else f"weather_{architecture}_all_open_weak_sources.pt")
    torch.save({"state_dict": model.state_dict(),
                "model": model_class.__name__, "feature_names": FEATURE_NAMES,
                "weather_names": WEATHER_NAMES, "normalization": norm,
                "trained_sources": list(sources),
                "excluded_regions": list(exclude_regions),
                "label_semantics": "relative mapped-water ranking, not calibrated flood probability"},
               checkpoint)
    report = {"trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "model": model_class.__name__,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "feature_names": FEATURE_NAMES, "weather_names": WEATHER_NAMES,
              "source_scene_counts": dict(Counter(s["source"] for s in scenes)),
              "excluded_regions": list(exclude_regions),
              "source_sampling_counts": dict(sampled),
              "distinct_regions": norm["distinct_regions"],
              "training_steps": EPOCHS * STEPS_PER_EPOCH,
              "curve": curve,
              "target": "source-weighted positive-area ranking plus low-weight UNOSAT likely-dry loss",
              "caution": "No held-out evaluation; daily rainfall may postdate some satellite acquisitions; outputs are uncalibrated scores"}
    (output_dir / "training_run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-bangkok", action="store_true")
    parser.add_argument("--model", choices=("unet", "cnn"), default="unet")
    args = parser.parse_args()
    excluded = ("nong_chok", "bang_kapi_core") if args.holdout_bangkok else ()
    destination = (OUTPUT / ("bangkok_holdout_cnn" if args.model == "cnn" else "bangkok_holdout")
                   if args.holdout_bangkok else OUTPUT)
    print(json.dumps(train(excluded, destination, args.model), indent=2))
