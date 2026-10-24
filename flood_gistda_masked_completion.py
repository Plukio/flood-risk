"""Masked GISTDA map completion with a separate no-observation rain-input test.

Only positive mapped-water area enters the loss. A missing GISTDA polygon is
unknown, not verified dry. Nong Chok never enters fitting or epoch selection.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from flood_bangkok_gistda_grid_backtest import selection_weights
from flood_gistda_multiregion_data import OUTPUT, SIZE
from flood_gistda_multiregion_train import TRAIN_DATES, VAL_DATES, density_loss, normalize, scenes
from flood_gistda_province_2025_extra_train import append_2025
from flood_gistda_province_expand_bangkok_eval import bma_arrays
from flood_gistda_province_expand_train import FEATURE_NAMES, TARGET, load_all


SEED = 94
BLOCK = 20  # 5 km; 16 nonoverlapping blocks in an 80 x 80 patch
CHECKPOINT = OUTPUT / "gistda_masked_completion_bangkok.pt"
RESULT = OUTPUT / "gistda_masked_completion_bangkok_metrics.json"


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CompletionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(len(FEATURE_NAMES) + 2, 16)
        self.enc2 = ConvBlock(16, 24)
        self.mid = ConvBlock(24, 32)
        self.dec2 = ConvBlock(32 + 24, 24)
        self.dec1 = ConvBlock(24 + 16, 16)
        self.out = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.log_crop_scale = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def forward(self, x: torch.Tensor, observed: torch.Tensor,
                known: torch.Tensor, crop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat((x, observed, known), dim=1)
        e1 = self.enc1(z)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        m = self.mid(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat((F.interpolate(m, size=e2.shape[-2:],
                                 mode="bilinear", align_corners=False), e2), dim=1))
        d1 = self.dec1(torch.cat((F.interpolate(d2, size=e1.shape[-2:],
                                 mode="bilinear", align_corners=False), e1), dim=1))
        residual = self.out(d1)[:, 0]
        prior = F.softplus(self.log_crop_scale) * torch.log(crop.clamp_min(0) + .03)
        return prior + residual, residual


def block_mask(row: int, col: int) -> np.ndarray:
    missing = np.zeros((SIZE, SIZE), bool)
    missing[row:row + BLOCK, col:col + BLOCK] = True
    return missing


def positive_blocks(wet: np.ndarray) -> list[tuple[int, int]]:
    return [(r, c) for r in range(0, SIZE, BLOCK)
            for c in range(0, SIZE, BLOCK)
            if wet[r:r + BLOCK, c:c + BLOCK].sum() > 1]


def model_input(wet: np.ndarray, missing: np.ndarray,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    known = (~missing).astype(np.float32)
    # Zero here means "no mapped polygon observed", never "verified dry".
    observed = np.clip(wet / 62500.0, 0, 1) * known
    return (torch.from_numpy(observed[None, None]).to(device),
            torch.from_numpy(known[None, None]).to(device))


def predict(model: CompletionUNet, x: np.ndarray, crop: np.ndarray,
            wet: np.ndarray, missing: np.ndarray, device: torch.device) -> np.ndarray:
    observed, known = model_input(wet, missing, device)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x[None]).to(device), observed, known,
                          torch.from_numpy(crop[None]).to(device))
    return logits[0].cpu().numpy()


def validation(model: CompletionUNet, regions: dict, pairs: list,
               crop: dict, device: torch.device) -> tuple[float, float]:
    model.eval()
    full, block = [], []
    with torch.no_grad():
        for name, di in pairs:
            item = regions[name]
            wet = item["wet"][di]
            x = torch.from_numpy(item["x"][di:di + 1]).to(device)
            prior = torch.from_numpy(crop[name][None]).to(device)
            zero = torch.zeros((1, 1, SIZE, SIZE), device=device)
            all_unknown = torch.zeros_like(zero)
            logits, _ = model(x, zero, all_unknown, prior)
            full.append(float(density_loss(logits[0], torch.from_numpy(wet).to(device))))
            blocks = positive_blocks(wet)
            # A fixed geographic block for each patch-day, independent of epoch.
            row, col = blocks[(di + sum(map(ord, name))) % len(blocks)]
            missing = block_mask(row, col)
            observed, known = model_input(wet, missing, device)
            logits, _ = model(x, observed, known, prior)
            truth = torch.from_numpy(wet[missing]).to(device)
            block.append(float(density_loss(logits[0][torch.from_numpy(missing).to(device)], truth)))
    return float(np.mean(full)), float(np.mean(block))


def fit(regions: dict, dates: list[str], crop: dict,
        train_regions: tuple[str, ...], device: torch.device) -> tuple[CompletionUNet, dict]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = CompletionUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    train_dates = TRAIN_DATES + tuple(range(8, len(dates)))
    train_pairs = scenes(regions, train_regions, train_dates)
    val_pairs = scenes(regions, train_regions, VAL_DATES)
    best = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    curve = []
    for epoch in range(1, 81):
        model.train()
        losses = []
        for idx in rng.permutation(len(train_pairs)):
            name, di = train_pairs[int(idx)]
            item = regions[name]
            wet = item["wet"][di]
            use_block = bool(rng.random() < .5)
            if use_block:
                blocks = positive_blocks(wet)
                row, col = blocks[int(rng.integers(len(blocks)))]
                missing = block_mask(row, col)
            else:
                missing = np.ones((SIZE, SIZE), bool)
            observed, known = model_input(wet, missing, device)
            x = torch.from_numpy(item["x"][di:di + 1]).to(device)
            prior = torch.from_numpy(crop[name][None]).to(device)
            logits, residual = model(x, observed, known, prior)
            mask = torch.from_numpy(missing).to(device)
            target = torch.from_numpy(wet[missing]).to(device)
            loss = density_loss(logits[0][mask], target) + .001 * residual.square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        full, blocked = validation(model, regions, val_pairs, crop, device)
        record = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
                  "val_full_loss": round(full, 5),
                  "val_block_loss": round(blocked, 5)}
        curve.append(record)
        selection = full + blocked
        if selection < best - .0005:
            best, best_epoch = selection, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == 1:
            print("epoch", record, "selected", best_epoch, flush=True)
        if epoch >= 25 and stale >= 15:
            break
    model.load_state_dict(best_state)
    return model, {"selected_epoch": best_epoch,
                   "positive_training_patch_days": len(train_pairs),
                   "positive_validation_patch_days": len(val_pairs),
                   "curve": curve}


def capture(score: np.ndarray, wet: np.ndarray, mask: np.ndarray) -> float:
    if wet[mask].sum() <= 0:
        return float("nan")
    return float(np.dot(selection_weights(score[mask], .1), wet[mask]) / wet[mask].sum())


def rain_doubled(x: np.ndarray, normalizer: dict) -> np.ndarray:
    """Double rain intensities, preserve wet-hour counts and other features."""
    changed = x.copy()
    mean = np.asarray(normalizer["mean"], np.float32)
    std = np.asarray(normalizer["std"], np.float32)
    for channel in (7, 8, 9, 11, 12, 13):
        raw = x[:, channel] * std[channel] + mean[channel]
        mm = np.expm1(raw).clip(min=0)
        doubled = np.log1p(2 * mm)
        changed[:, channel] = np.clip((doubled - mean[channel]) / std[channel], -5, 5)
    return changed


def evaluate(model: CompletionUNet, target: dict, dates: list[str],
             crop: np.ndarray, normalizer: dict, device: torch.device) -> dict:
    area, bma_wet = bma_arrays(dates[:8])
    bma = area > 1000
    results = {"full_unknown": [], "masked_blocks": [], "rain_sensitivity": []}
    for di, day in enumerate(dates[:8]):
        x = target["x"][di]
        wet = target["wet"][di]
        full_missing = np.ones((SIZE, SIZE), bool)
        score = predict(model, x, crop, wet, full_missing, device)
        doubled = predict(model, rain_doubled(x[None], normalizer)[0], crop,
                          wet, full_missing, device)
        result = {"date": day, "mapped_area_km2": float(bma_wet[di][bma].sum()/1e6),
                  "unet_capture": capture(score, bma_wet[di], bma),
                  "crop_capture": capture(crop, bma_wet[di], bma)}
        results["full_unknown"].append(result)
        selected = selection_weights(score[bma], .1) > 0
        selected_double = selection_weights(doubled[bma], .1) > 0
        union = np.logical_or(selected, selected_double).sum()
        results["rain_sensitivity"].append({"date": day,
            "top_10pct_jaccard_doubled_rain": float(np.logical_and(selected, selected_double).sum()/union),
            "cell_spearman_doubled_rain": float(spearmanr(score[bma], doubled[bma]).statistic)})
        for row in range(0, SIZE, BLOCK):
            for col in range(0, SIZE, BLOCK):
                missing = block_mask(row, col)
                test = bma & missing
                if test.sum() < 10 or bma_wet[di][test].sum() <= 1:
                    continue
                completed = predict(model, x, crop, wet, missing, device)
                results["masked_blocks"].append({"date": day, "row": row,
                    "col": col, "cells": int(test.sum()),
                    "mapped_area_km2": float(bma_wet[di][test].sum()/1e6),
                    "unet_capture": capture(completed, bma_wet[di], test),
                    "crop_capture": capture(crop, bma_wet[di], test)})
        print("eval", day, result, flush=True)
    for key in ("full_unknown", "masked_blocks"):
        rows = results[key]
        weight = sum(r["mapped_area_km2"] for r in rows)
        results[key + "_summary"] = {"evaluations": len(rows),
            "mapped_area_days_km2": round(weight, 4),
            **{name: round(sum(r[name]*r["mapped_area_km2"] for r in rows)/weight, 5)
               for name in ("unet_capture", "crop_capture")}}
    return results


def main() -> None:
    torch.set_num_threads(4)
    device = torch.device("cpu")
    regions, dates = load_all()
    # Save raw WorldCover crop fraction before feature normalization.
    crop = {name: item["x"][0, 17].copy() for name, item in regions.items()}
    dates = append_2025(regions, dates)
    train_regions = tuple(name for name in regions if name != TARGET)
    normalizer = normalize(regions, train_regions)
    model, training = fit(regions, dates, crop, train_regions, device)
    torch.save({"state_dict": model.cpu().state_dict(),
                "train_regions": train_regions,
                "train_dates": [dates[i] for i in TRAIN_DATES + tuple(range(8, len(dates)))],
                "feature_names": FEATURE_NAMES, "normalization": normalizer,
                "selected_epoch": training["selected_epoch"]}, CHECKPOINT)
    results = evaluate(model, regions[TARGET], dates, crop[TARGET], normalizer, device)
    report = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model": "crop-prior U-Net, positive-area density loss",
              "target": "GISTDA mapped positive water, not dry/wet classification",
              "test_region": TARGET,
              "training_regions": train_regions,
              "normalization": normalizer,
              "training": training,
              "masking": "Half training samples hide a 20x20-cell block; half hide the complete 80x80 map. At block evaluation, same-day GISTDA mapped positives outside the hidden block are supplied as context.",
              "rain_caveat": "Doubled-rain output is sensitivity only, not a validated counterfactual; full-day rain may postdate the image.",
              **results}
    RESULT.write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY", results["full_unknown_summary"],
          results["masked_blocks_summary"], flush=True)


if __name__ == "__main__":
    main()
