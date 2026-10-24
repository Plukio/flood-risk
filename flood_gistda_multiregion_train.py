"""Train positive-only GISTDA spatial rankings with regional/date holdouts.

Loss is conditional density over each patch's cells: known mapped-water area
is target mass. Absence of GISTDA polygons is never a dry label.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from flood_bangkok_gistda_grid_backtest import selection_weights
from flood_gistda_multiregion_data import CACHE, OUTPUT, CENTERS, SIZE
from flood_gistda_multiregion_sentinel2 import CHANNELS as SENTINEL_NAMES


SEED = 71
TRAIN_DATES = (0, 1, 2, 3, 4)  # 27 Sep, 2/3/4/7 Oct 2026
VAL_DATES = (5, 6)            # 9/10 Oct 2026
TEST_DATE = 7                 # 20 Oct 2026
STATIC_NAMES = ("srtm_elevation_m", "srtm_relative_1250m_m", "srtm_slope",
                "osm_canal_length_log", "osm_canal_distance_log",
                "osm_road_length_log", "osm_road_distance_log")
RAIN_NAMES = ("rain_today_mm_log", "rain_peak_1h_mm_log", "rain_peak_3h_mm_log",
              "rain_wet_hours_log", "rain_prev24h_mm_log", "rain_prev72h_mm_log",
              "rain_today_prev24h_mm_log")
BASE_FEATURE_NAMES = STATIC_NAMES + RAIN_NAMES + ("month_sin", "month_cos")


class SpatialRankCNN(nn.Module):
    def __init__(self, n_features: int, linear: bool = False):
        super().__init__()
        self.layers = (nn.Conv2d(n_features, 1, 1) if linear else
                       nn.Sequential(nn.Conv2d(n_features, 24, 3, padding=1),
                                     nn.ReLU(), nn.Conv2d(24, 16, 3, padding=1),
                                     nn.ReLU(), nn.Conv2d(16, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)[:, 0]


def load_data(with_sentinel: bool) -> tuple[dict, list[str], tuple[str, ...]]:
    regions = {}
    dates = None
    for name in CENTERS:
        with np.load(CACHE / f"{name}_wet_250m.npz") as saved:
            wet = saved["wet_area_m2"].astype(np.float32)
            current_dates = list(saved["dates"].astype(str))
        if dates is None:
            dates = current_dates
        elif dates != current_dates:
            raise ValueError("GISTDA dates differ between regions")
        with np.load(CACHE / f"{name}_static_250m.npz") as saved:
            static = saved["values"].astype(np.float32)
        with np.load(CACHE / f"{name}_rain_features_2026.npz") as saved:
            rain = saved["values"].astype(np.float32)
            if list(saved["dates"].astype(str)) != dates:
                raise ValueError("Rain dates differ from GISTDA dates")
        if wet.shape != (8, SIZE, SIZE) or static.shape != (7, SIZE, SIZE) or rain.shape != (8, 7):
            raise ValueError(f"Feature shape changed: {name}")
        months = np.asarray([int(d[5:7]) for d in dates])
        global_values = np.concatenate((rain,
                                        np.sin(2 * np.pi * months[:, None] / 12),
                                        np.cos(2 * np.pi * months[:, None] / 12)), axis=1)
        x = np.concatenate((np.broadcast_to(static, (8, 7, SIZE, SIZE)),
                            np.broadcast_to(global_values[:, :, None, None],
                                            (8, 9, SIZE, SIZE))), axis=1).copy()
        if with_sentinel:
            with np.load(CACHE / f"{name}_sentinel2_2025_250m.npz") as saved:
                extra = saved["values"].astype(np.float32)
                if tuple(saved["names"].astype(str)) != SENTINEL_NAMES:
                    raise ValueError(f"Sentinel-2 channels differ: {name}")
            if extra.shape != (3, SIZE, SIZE):
                raise ValueError(f"Sentinel-2 shape changed: {name}")
            x = np.concatenate((x, np.broadcast_to(extra, (8, 3, SIZE, SIZE))), axis=1).copy()
        regions[name] = {"x": x, "wet": wet, "static": static}
    feature_names = BASE_FEATURE_NAMES + (SENTINEL_NAMES if with_sentinel else ())
    return regions, dates, feature_names


def normalize(regions: dict, train_regions: tuple[str, ...]) -> dict:
    samples = np.concatenate([regions[name]["x"][list(TRAIN_DATES)]
                              for name in train_regions], axis=0)
    mean = samples.mean(axis=(0, 2, 3))
    std = np.maximum(samples.std(axis=(0, 2, 3)), .02)
    for item in regions.values():
        item["x"] = np.clip((item["x"] - mean[None, :, None, None]) /
                            std[None, :, None, None], -5, 5).astype(np.float32)
    return {"mean": mean.tolist(), "std": std.tolist()}


def scenes(regions: dict, names: tuple[str, ...], date_ids: tuple[int, ...]) -> list:
    return [(name, i) for name in names for i in date_ids
            if regions[name]["wet"][i].sum() > 1]


def density_loss(logits: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
    target = wet.ravel() / wet.sum()
    return -(target * F.log_softmax(logits.ravel(), dim=0)).sum()


def capture(scores: np.ndarray, wet: np.ndarray) -> dict:
    target = wet.ravel().astype(float)
    if target.sum() <= 0:
        raise ValueError("Cannot evaluate a day without mapped water")
    return {f"top_{int(f*100)}pct_area_capture": round(float(
        np.dot(selection_weights(scores.ravel(), f), target) / target.sum()), 5)
            for f in (.05, .10, .20)}


def evaluate(model: nn.Module, regions: dict, pairs: list,
             dates: list[str], reference: str) -> list[dict]:
    model.eval()
    output = []
    with torch.no_grad():
        for name, i in pairs:
            x = torch.from_numpy(regions[name]["x"][i:i+1])
            logits = model(x).numpy()[0]
            wet = regions[name]["wet"][i]
            row = {"model": reference, "region": name, "date": dates[i],
                   "mapped_wet_area_km2": round(float(wet.sum()) / 1e6, 5),
                   "mapped_wet_cells": int(np.count_nonzero(wet > 1)),
                   **capture(logits, wet)}
            output.append(row)
    return output


def train_one(kind: str, regions: dict, dates: list[str],
              train_regions: tuple[str, ...], feature_names: tuple[str, ...],
              run_name: str, train_dates: tuple[int, ...] = TRAIN_DATES) -> tuple[nn.Module, dict]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = SpatialRankCNN(len(feature_names), linear=(kind == "linear"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002,
                                  weight_decay=.02)
    train_pairs = scenes(regions, train_regions, train_dates)
    val_pairs = scenes(regions, train_regions, VAL_DATES)
    if len(train_pairs) < 8 or len(val_pairs) < 3:
        raise ValueError("Not enough positive patch-days for training/validation")
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    curve = []
    for epoch in range(1, 121):
        model.train()
        losses = []
        for idx in rng.permutation(len(train_pairs)):
            name, i = train_pairs[int(idx)]
            x = torch.from_numpy(regions[name]["x"][i:i+1])
            wet = torch.from_numpy(regions[name]["wet"][i])
            optimizer.zero_grad()
            logits = model(x)[0]
            loss = density_loss(logits, wet) + .0001 * logits.square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val = []
            for name, i in val_pairs:
                x = torch.from_numpy(regions[name]["x"][i:i+1])
                wet = torch.from_numpy(regions[name]["wet"][i])
                val.append(float(density_loss(model(x)[0], wet)))
        record = {"epoch": epoch, "train_density_loss": round(float(np.mean(losses)), 5),
                  "val_density_loss": round(float(np.mean(val)), 5)}
        curve.append(record)
        if record["val_density_loss"] < best - .0001:
            best, best_epoch = record["val_density_loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 10 == 0 or epoch == 1:
            print(run_name, kind, record, "best", best_epoch, flush=True)
        if epoch >= 25 and stale >= 18:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "feature_names": feature_names,
                "train_regions": train_regions, "train_dates": [dates[i] for i in train_dates],
                "selected_epoch": best_epoch, "positive_only": True},
               OUTPUT / f"gistda_{run_name}_{kind}_spatial_rank.pt")
    return model, {"selected_epoch": best_epoch, "best_val_density_loss": best,
                   "train_patch_days": len(train_pairs), "val_patch_days": len(val_pairs),
                   "curve": curve}


def baselines(regions: dict, pairs: list, dates: list[str],
              train_regions: tuple[str, ...]) -> list[dict]:
    rows = []
    for name, i in pairs:
        wet = regions[name]["wet"][i]
        static = regions[name]["static"]
        ranks = {"random_expected": np.zeros_like(wet),
                 "low_elevation": -static[0],
                 "near_canal": -static[4],
                 "osm_road_density": static[5]}
        if name in train_regions and i == TEST_DATE:
            ranks["gistda_history_train_dates"] = regions[name]["wet"][list(TRAIN_DATES)].sum(axis=0)
            ranks["gistda_history_through_oct10"] = regions[name]["wet"][:TEST_DATE].sum(axis=0)
        for model_name, score in ranks.items():
            rows.append({"model": model_name, "region": name, "date": dates[i],
                         "mapped_wet_area_km2": round(float(wet.sum()) / 1e6, 5),
                         "mapped_wet_cells": int(np.count_nonzero(wet > 1)),
                         **capture(score, wet)})
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["model"], row["split"])
        groups.setdefault(key, []).append(row)
    result = []
    for (model, split), items in groups.items():
        area = sum(x["mapped_wet_area_km2"] for x in items)
        result.append({"model": model, "split": split, "positive_patch_days": len(items),
                       "mapped_area_days_km2": round(area, 4),
                       **{key: round(sum(x[key] * x["mapped_wet_area_km2"] for x in items)
                                     / area, 5)
                          for key in ("top_5pct_area_capture", "top_10pct_area_capture",
                                      "top_20pct_area_capture")}})
    return sorted(result, key=lambda x: (x["split"], x["model"]))


def learning_plot(training: dict) -> None:
    rows = math.ceil(len(training) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(15, rows * 3), dpi=150, sharey=True)
    for ax, (name, data) in zip(axes.ravel(), training.items()):
        curve = data["curve"]
        ax.plot([x["epoch"] for x in curve], [x["train_density_loss"] for x in curve],
                label="fit")
        ax.plot([x["epoch"] for x in curve], [x["val_density_loss"] for x in curve],
                label="later-date validation")
        ax.axvline(data["selected_epoch"], color="k", linestyle=":", linewidth=.8)
        ax.set_title(name.replace("_holdout", ""), fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Positive-area density loss")
        ax.legend(fontsize=6)
    for ax in axes.ravel()[len(training):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT / "gistda_multiregion_learning_curves.png")
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    training = {}
    evaluation = []
    normalization = {}
    for holdout in CENTERS:
        train_regions = tuple(name for name in CENTERS if name != holdout)
        for with_sentinel, kind in ((False, "cnn"), (True, "cnn")):
            regions, dates, feature_names = load_data(with_sentinel)
            run_name = f"holdout_{holdout}_{'s2' if with_sentinel else 'base'}"
            normalization[run_name] = normalize(regions, train_regions)
            model, training[run_name] = train_one(
                kind, regions, dates, train_regions, feature_names, run_name)
            model_name = "cnn_s2" if with_sentinel else "cnn_base"
            geographic = scenes(regions, (holdout,), tuple(range(len(dates))))
            temporal = scenes(regions, train_regions, (TEST_DATE,))
            if not geographic or not temporal:
                raise ValueError(f"No positive GISTDA test scenes: {holdout}")
            for split, pairs in (("geographic_holdout", geographic),
                                 ("later_date_test", temporal)):
                rows = evaluate(model, regions, pairs, dates, model_name)
                for row in rows:
                    row["split"] = split
                    row["holdout_region"] = holdout
                evaluation.extend(rows)
            if not with_sentinel:
                rows = baselines(regions, geographic, dates, train_regions)
                for row in rows:
                    row["split"] = "geographic_holdout"
                    row["holdout_region"] = holdout
                evaluation.extend(rows)
                rows = baselines(regions, temporal, dates, train_regions)
                for row in rows:
                    row["split"] = "later_date_test"
                    row["holdout_region"] = holdout
                evaluation.extend(rows)
    fold_summary = []
    for holdout in CENTERS:
        fold_rows = [x for x in evaluation if x["split"] == "geographic_holdout"
                     and x["holdout_region"] == holdout]
        for row in aggregate(fold_rows):
            row["holdout_region"] = holdout
            fold_summary.append(row)
    summary = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "target": "conditional spatial density of GISTDA mapped-water area in a fixed patch on a product day",
               "not_target": "verified dry/wet classification, flood probability or road flood",
               "feature_names_base": BASE_FEATURE_NAMES,
               "feature_names_sentinel": BASE_FEATURE_NAMES + SENTINEL_NAMES,
               "normalization_from_training": normalization,
               "geographic_holdouts": list(CENTERS),
               "train_dates": [dates[i] for i in TRAIN_DATES],
               "selection_dates": [dates[i] for i in VAL_DATES],
               "later_date_test": dates[TEST_DATE],
               "loss": "negative area-weighted log softmax over all 6400 cells within each patch; unlabeled cells are alternatives in a conditional density, not verified dry",
               "training": training, "summary": aggregate(evaluation),
               "geographic_fold_summary": fold_summary,
               "limitations": ["Patch centers selected after checking GISTDA mapped-water coverage; geographic selection is exploratory",
                               "Only eight related 2026 product dates; repeated mapped water creates strong temporal autocorrelation",
                               "No GISTDA acquisition/observed-dry mask; scores rank mapped-water occurrence conditional on product coverage",
                               "GISTDA labels favor broad eastern/agricultural water, not urban road flood",
                               "SRTM 30 m terrain is lower quality than BMA LiDAR in flat terrain",
                               "One Open-Meteo grid point per 20 km patch; no water-level or pump/gate state",
                               "Whole-day rainfall may include hours after satellite acquisition because GISTDA polygon acquisition times are not confirmed",
                               "Sentinel-2 2025 NDVI is a vegetation proxy, not soil infiltration; 2025 land cover may differ in 2026"]}
    (OUTPUT / "gistda_multiregion_training_metrics.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2) + "\n")
    with (OUTPUT / "gistda_multiregion_evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(evaluation[0]))
        writer.writeheader()
        writer.writerows(evaluation)
    learning_plot(training)
    print("saved training metrics", OUTPUT / "gistda_multiregion_training_metrics.json", flush=True)
    for row in summary["summary"]:
        print(row, flush=True)


if __name__ == "__main__":
    main()
