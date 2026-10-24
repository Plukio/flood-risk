"""Compare spatial U-Nets with district-wide weather at input vs bottleneck.

Both models use the same 250 m cells, 42 training districts, eight geographic
holdouts, chronological split, normalized features, seed, optimizer, and
training schedule. The outcome is Traffy complaint occurrence, not flooding.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from flood_bangkok_scenario_stress import SCENARIOS, scene, weather
from flood_bangkok_spatial_cnn_citywide import (
    BATCH, CAL_N, CHANNELS, HOLDOUT, POSITIVE_WEIGHT, SEED, TRAIN_N,
    evaluate, prepare_cache, transform,
)
from flood_bangkok_spatial_cnn_infrastructure import augmented_arrays
from flood_bangkok_transfer import load_districts
from flood_bangkapi_spatial_cnn import SpatialCNN
from flood_bangkapi_surface_model import OUTPUT, write_csv


ARTIFACT = "surface_crossdistrict_unet_global_bottleneck_v1"
MASKED = tuple(range(CHANNELS + 2, CHANNELS + 7))
MODES = ("broadcast", "bottleneck")
STEPS_PER_EPOCH = 400
MAX_EPOCHS = 25
PATIENCE = 8


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GlobalUNet(nn.Module):
    """The same parameters for either global-input or global-bottleneck use."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        # 8 base spatial features (terrain/assets/history), 2 canal geometry,
        # valid-cell mask, and 9 slots for season/rain summaries.
        self.encoder1 = ConvBlock(20, 16)
        self.encoder2 = ConvBlock(16, 24)
        self.bottleneck = ConvBlock(24, 32)
        self.condition = nn.Sequential(nn.Linear(9, 32), nn.ReLU(),
                                       nn.Linear(32, 64))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        self.decoder2 = ConvBlock(32 + 24, 24)
        self.decoder1 = ConvBlock(24 + 16, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, -1:]
        spatial = torch.cat((x[:, :8], x[:, CHANNELS:CHANNELS + 2], mask), dim=1)
        global_map = x[:, 8:CHANNELS]
        global_vector = (global_map * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1)
        if self.mode == "bottleneck":
            encoder_global = torch.zeros_like(global_map)
            condition_global = global_vector
        else:
            encoder_global = global_map
            condition_global = torch.zeros_like(global_vector)
        z = torch.cat((spatial, encoder_global), dim=1)
        e1 = self.encoder1(z)
        e2 = self.encoder2(F.avg_pool2d(e1, 2, ceil_mode=True))
        b = self.bottleneck(F.avg_pool2d(e2, 2, ceil_mode=True))
        gamma, beta = self.condition(condition_global).chunk(2, dim=1)
        b = b * (1 + gamma.tanh()[:, :, None, None]) + beta[:, :, None, None]
        d2 = self.decoder2(torch.cat((F.interpolate(
            b, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2), dim=1))
        d1 = self.decoder1(torch.cat((F.interpolate(
            d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), dim=1))
        return self.output(d1)


@torch.no_grad()
def predict(model: GlobalUNet, raw: np.ndarray, stats: list,
            rr: np.ndarray, cc: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(raw), BATCH):
        x = torch.from_numpy(transform(raw[start:start + BATCH], stats, MASKED)).to(device)
        out.append(expit(model(x).cpu().numpy()[:, 0, rr, cc]))
    return np.concatenate(out)


def train(mode: str, codes: tuple[str, ...], arrays: dict, meta: dict,
          stats: list, device: torch.device, artifact: str,
          steps_per_epoch: int, max_epochs: int, patience: int):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = GlobalUNet(mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    bk_x, bk_y = arrays["1006"]
    rr = np.asarray(meta["1006"]["valid_rows"])
    cc = np.asarray(meta["1006"]["valid_cols"])
    truth = np.asarray(bk_y[TRAIN_N:CAL_N, 0, rr, cc]).ravel()
    best_ap, best_epoch, best_state, stale = -1.0, 0, None, 0
    curve = []
    positive_weight = torch.tensor(POSITIVE_WEIGHT, device=device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            code = codes[int(rng.integers(len(codes)))]
            raw_x, raw_y = arrays[code]
            ids = rng.integers(0, TRAIN_N, size=BATCH)
            x = torch.from_numpy(transform(raw_x[ids], stats, MASKED)).to(device)
            y = torch.from_numpy(np.asarray(raw_y[ids], dtype=np.float32)).to(device)
            mask = x[:, -1:]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(x), y, pos_weight=positive_weight, reduction="none")
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = predict(model, bk_x[TRAIN_N:CAL_N], stats, rr, cc, device).ravel()
        ap = float(average_precision_score(truth, score))
        curve.append({"epoch": epoch, "train_loss_mean": round(float(np.mean(losses)), 6),
                      "bang_kapi_cal_ap": round(ap, 6)})
        if ap > best_ap + 1e-6:
            best_ap, best_epoch = ap, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"{mode} epoch {epoch}: loss={curve[-1]['train_loss_mean']:.5f} "
              f"Bang Kapi AP={ap:.5f} best={best_ap:.5f}", flush=True)
        if stale >= patience and epoch >= 10:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "stats": stats, "training_codes": codes,
                "selected_epoch": best_epoch, "mode": mode},
               OUTPUT / f"{artifact}_{mode}.pt")
    return model, {"training_codes": list(codes), "feature_channels": CHANNELS + 7,
                   "masked_channels": list(MASKED),
                   "district_days_sampled_per_epoch": BATCH * steps_per_epoch,
                   "maximum_epochs": max_epochs, "early_stop_patience": patience,
                   "selected_epoch": best_epoch,
                   "selection_bang_kapi_cal_ap": round(best_ap, 6),
                   "curve": curve, "device": str(device)}


def scenario_response(models: dict, stats: list, arrays: dict,
                      metadata: dict, rows: dict, device: torch.device,
                      artifact: str) -> dict:
    result = {}
    district = []
    for mode, model in models.items():
        totals = {name: 0.0 for name in SCENARIOS}
        decreases = {name: 0 for name in SCENARIOS}
        rank_correlations = {name: [] for name in SCENARIOS}
        for code, (x, _) in arrays.items():
            rr = np.asarray(metadata[code]["valid_rows"])
            cc = np.asarray(metadata[code]["valid_cols"])
            predictions = {}
            raw_day = np.asarray(x[CAL_N:CAL_N + 1], dtype=np.float32)
            for name, config in SCENARIOS.items():
                perturbed = scene(raw_day, weather(*config))
                normalized = torch.from_numpy(transform(perturbed, stats, MASKED)).to(device)
                with torch.no_grad():
                    logits = model(normalized).cpu().numpy()[0, 0, rr, cc]
                predictions[name] = expit(logits + rows[mode][code]["log_odds_shift"])
            base = predictions["20mm_2h"]
            for name, values in predictions.items():
                total = float(values.sum())
                totals[name] += total
                decreases[name] += int(np.count_nonzero(values < base - 1e-9))
                correlation = float(spearmanr(values, base).statistic)
                rank_correlations[name].append(correlation)
                district.append({"mode": mode, "code": code, "scenario": name,
                                 "cells": len(rr), "sum_proxy_score": round(total, 5),
                                 "score_index_vs_20mm_2h": round(total / base.sum(), 5),
                                 "rank_spearman_vs_20mm_2h": round(correlation, 5),
                                 "cells_decreasing_vs_20mm_2h": int(np.count_nonzero(
                                     values < base - 1e-9))})
        total_cells = sum(len(metadata[code]["valid_rows"]) for code in arrays)
        result[mode] = {name: {
            "score_index_vs_20mm_2h": round(totals[name] / totals["20mm_2h"], 5),
            "fraction_cells_decreasing_vs_20mm_2h": round(decreases[name] / total_cells, 5),
            "mean_district_rank_spearman_vs_20mm_2h": round(float(np.mean(
                rank_correlations[name])), 5)} for name in SCENARIOS}
    write_csv(OUTPUT / f"{artifact}_scenario_district.csv", district)
    return result


def main(matched_long: bool = False) -> None:
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    districts = {d["code"]: d for d in load_districts()}
    meta = prepare_cache(districts)
    arrays = augmented_arrays(meta)
    if int(np.asarray(arrays["1006"][1][CAL_N:]).sum()) != 242:
        raise ValueError("Bang Kapi labels changed")
    codes = tuple(code for code in districts if code not in HOLDOUT)
    artifact = ARTIFACT + ("_matched_long" if matched_long else "")
    steps = 900 if matched_long else STEPS_PER_EPOCH
    max_epochs = 40 if matched_long else MAX_EPOCHS
    patience = 18 if matched_long else PATIENCE
    canal_ck = torch.load(
        OUTPUT / "surface_crossdistrict_infrastructure_ablation_citywide_v1_canal_only.pt",
        map_location="cpu", weights_only=False)
    stats = canal_ck["stats"]
    if len(stats) != CHANNELS + 7 or tuple(canal_ck["codes"]) != codes:
        raise ValueError("Canal CNN reference inputs/training codes changed")
    models, results, by_code = {}, {}, {}
    for mode in MODES:
        model, training = train(mode, codes, arrays, meta, stats, device,
                                artifact, steps, max_epochs, patience)
        rows = evaluate(model, stats, arrays, meta, tuple(districts),
                        masked_channels=MASKED)
        models[mode] = model
        results[mode] = {"training": training, "evaluated_districts": rows}
        by_code[mode] = {r["code"]: r for r in rows}
    reference = json.loads((OUTPUT / "surface_crossdistrict_infrastructure_ablation_citywide_v1_metrics.json").read_text())[
        "models"]["canal_only"]["evaluated_districts"]
    old = {r["code"]: r for r in reference}
    comparison = []
    for code, district in districts.items():
        row = {"code": code, "district": district["name_en"],
               "group": "geographic_holdout" if code in HOLDOUT else "training_district",
               "test_positive_cell_days": old[code]["test_positive_cell_days"],
               "historical_spearman": old[code]["historical_spearman"],
               "canal_cnn_spearman": old[code]["cnn_spearman"],
               "canal_cnn_daily_ap": old[code]["test_daily_ap"]}
        for mode in MODES:
            item = by_code[mode][code]
            if (item["test_positive_cell_days"] != old[code]["test_positive_cell_days"] or
                    item["cells"] != old[code]["cells"]):
                raise ValueError(f"Outcome/grid changed for {code}")
            row[f"{mode}_spearman"] = item["cnn_spearman"]
            row[f"{mode}_daily_ap"] = item["test_daily_ap"]
            row[f"{mode}_predicted_positive_cell_days"] = item[
                "predicted_positive_cell_days"]
        comparison.append(row)
    write_csv(OUTPUT / f"{artifact}_district_comparison.csv", comparison)
    scenarios = scenario_response(models, stats, arrays, meta, by_code, device,
                                  artifact)
    payload = {"target": "100 m direct Traffy flood-complaint buffer touches a 250 m cell-day",
               "not_target": "verified flood extent or depth",
               "architecture": "two-level U-Net, 8 base spatial channels, 2 canal channels, mask; 9 district-wide season/rain channels at input or FiLM bottleneck",
               "matched_comparison": "same model parameter shapes, masks, inputs, normalization, training schedule and seed; only placement of global information differs",
               "matched_long_schedule": matched_long,
               "train_period": ["2022-01-01", "2025-07-31"],
               "selection_and_local_calibration": ["2025-08-01", "2025-09-30"],
               "test_period": ["2025-10-01", "2025-12-31"],
               "geographic_holdout_codes": list(HOLDOUT),
               "scenarios": scenarios, "models": results,
               "limitations": ["district-wide rain is one gridded estimate repeated across cells",
                               "BMA road floods and verified dry cells are not training labels",
                               "October–December 2025 has been reused across experiments",
                               "synthetic rain-response is not verified flood simulation",
                               "training exposure differs from the saved older CNN baseline" if not matched_long else
                               "architecture and selection differ from the saved older CNN baseline"]}
    (OUTPUT / f"{artifact}_metrics.json").write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-long", action="store_true",
                        help="Match the prior citywide CNN's 900 steps, 40-epoch maximum and patience 18")
    main(parser.parse_args().matched_long)
