"""Matched local and transfer ablation of four open-data feature groups.

The outcome is a 100 m Traffy report buffer touching a 250 m cell-day, not
observed inundation. All variants use identical sampled rows and tree settings.

Run: .venv/bin/python flood_bangkok_new_feature_ablation.py
"""

from __future__ import annotations

import csv
import json
from datetime import date

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from flood_bangkok_spatial_water_state import (
    BASE_FEATURES, CODES, load_district, matrix, model, training_indices,
)
from flood_bangkok_river_level_backtest import river_features, river_levels
from flood_bangkok_spatial_cnn_multiyear import TRAIN_END
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import OUTPUT, TO_UTM, cell_box, grid_spec
from flood_bangkapi_surface_model import rainfall_by_date, rank_metrics


HII_FILE = CACHE / "hii_rain_daily_2022_2025.csv"
STATION_FILE = CACHE.parent / "flood_road_pilot" / "hii_station_metadata.csv"
ARTIFACT = "surface_new_feature_group_ablation"
MAX_GAUGE_DISTANCE_M = 20_000
IDW_SOFTEN_M = 2_000
GROUPS = {
    "terrain": (0, 1, 2, 3, 4),
    "surface_cover": (5, 6, 7, 8, 9, 10),
    "canal_graph": (0, 1, 2, 3, 4, 5),
}
RAIN_NAMES = (
    "hii_local_daily_rain_log", "hii_local_peak_1h_log",
    "hii_nearest_valid_station_km", "hii_valid_stations_within_20km",
    "hii_missing_within_20km",
)
VARIANTS = ("baseline", "+terrain", "+surface_cover", "+canal_graph",
            "+hii_rain", "+all")


def station_locations() -> dict[str, tuple[float, float]]:
    result = {}
    with STATION_FILE.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            try:
                lat, lon = float(row["Latitude"]), float(row["Longitude"])
            except (ValueError, TypeError):
                continue
            if 13.2 < lat < 14.3 and 100.0 < lon < 101.0:
                result[row["Station_Code"]] = TO_UTM.transform(lon, lat)
    return result


def rain_tables(days: list[date]) -> tuple[list[str], np.ndarray, np.ndarray]:
    positions = station_locations()
    by_day = {str(day): i for i, day in enumerate(days)}
    codes = []
    measurements = {}
    with HII_FILE.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            code = row["station_code"]
            if code not in positions or row["complete"] != "1":
                continue
            if code not in measurements:
                codes.append(code)
                measurements[code] = np.full((len(days), 2), np.nan, dtype=np.float32)
            di = by_day.get(row["date"])
            if di is not None:
                measurements[code][di] = float(row["rain_mm"]), float(row["peak_1h_mm"])
    codes = sorted(codes)
    return codes, np.asarray([positions[code] for code in codes]), np.stack(
        [measurements[code] for code in codes], axis=1)


def spatial_rain(code: str, days: list[date], stations: list[str],
                 locations: np.ndarray, rain: np.ndarray, rr: np.ndarray,
                 cc: np.ndarray) -> tuple[np.ndarray, dict]:
    district = next(d for d in load_districts() if d["code"] == code)
    grid = grid_spec(district["polygon"])
    points = np.asarray([(cell_box(grid, int(r), int(c)).centroid.x,
                          cell_box(grid, int(r), int(c)).centroid.y)
                         for r, c in zip(rr, cc)])
    distance = np.sqrt(((points[:, None, :] - locations[None, :, :]) ** 2).sum(axis=2))
    local = distance <= MAX_GAUGE_DISTANCE_M
    within = local.sum(axis=1)
    result = np.full((len(days), len(rr), len(RAIN_NAMES)), np.nan, dtype=np.float32)
    result[:, :, 3] = 0
    result[:, :, 4] = 1
    nearest = np.full((len(days), len(rr)), np.nan, dtype=np.float32)
    for di in range(len(days)):
        valid = np.isfinite(rain[di, :, 0])
        usable = local & valid[None, :]
        n = usable.sum(axis=1)
        result[di, :, 3] = n
        result[di, :, 4] = n == 0
        any_station = n > 0
        if not any_station.any():
            continue
        nearest[di, any_station] = np.where(
            usable[any_station], distance[any_station], np.inf).min(axis=1) / 1000
        result[di, :, 2] = nearest[di]
        weight = np.where(usable, 1 / (distance**2 + IDW_SOFTEN_M**2), 0)
        denominator = weight.sum(axis=1)
        for variable in range(2):
            observed = np.where(valid, rain[di, :, variable], 0)
            estimated = np.divide(weight @ observed, denominator,
                                  out=np.full(len(rr), np.nan), where=denominator > 0)
            result[di, :, variable] = np.log1p(estimated)
    # This is a spatial interpolation from actual complete station-days, not
    # synthetic street-scale rain. A lone station gives no spatial gradient.
    counts = result[:, :, 3]
    wet = result[:, :, 0] > np.log1p(1)
    wet_day_values = np.where(wet[np.any(wet, axis=1)],
                              result[np.any(wet, axis=1), :, 0], np.nan)
    spatial_range = (np.nanmax(wet_day_values, axis=1) -
                     np.nanmin(wet_day_values, axis=1))
    audit = {
        "stations_with_valid_days": stations,
        "cells_with_at_least_one_station_within_20km": int((within > 0).sum()),
        "cell_days_with_valid_station_fraction": round(float((counts > 0).mean()), 4),
        "cell_days_with_two_valid_stations_fraction": round(float((counts > 1).mean()), 4),
        "q4_2025_cell_days_with_valid_station_fraction": round(float((
            counts[np.asarray(days) >= date(2025, 10, 1)] > 0).mean()), 4),
        "nearest_valid_station_km_median": round(float(np.nanmedian(nearest)), 2),
        "wet_day_within_district_log_rain_range_median": round(float(np.median(spatial_range)), 3)
        if len(spatial_range) else None,
    }
    return result, audit


def load_group(code: str, kind: str, rr: np.ndarray, cc: np.ndarray) -> tuple[np.ndarray, list[str]]:
    with np.load(CACHE / f"{kind}_{code}_250m.npz") as saved:
        if not (np.array_equal(rr, saved["rows"]) and np.array_equal(cc, saved["cols"])):
            raise ValueError(f"Cell order differs in {kind} {code}")
        return saved["values"], list(saved["names"].astype(str))


def feature_matrix(info: dict, di: np.ndarray, ci: np.ndarray,
                   variant: str) -> np.ndarray:
    parts = [matrix(info, di, ci, "baseline")]
    if variant in ("+terrain", "+all"):
        parts.append(info["new_static"][ci][:, GROUPS["terrain"]])
    if variant in ("+surface_cover", "+all"):
        parts.append(info["new_static"][ci][:, GROUPS["surface_cover"]])
    if variant in ("+canal_graph", "+all"):
        parts.append(info["canal_graph"][ci][:, GROUPS["canal_graph"]])
    if variant in ("+hii_rain", "+all"):
        parts.append(info["spatial_rain"][di, ci])
    return np.column_stack(parts).astype(np.float32)


def evaluate_fold(data: dict, days: list[date], train_end: date,
                  test_start: date, test_end: date) -> dict:
    n_train = sum(day <= train_end for day in days)
    n_test_start = days.index(test_start)
    n_test_end = days.index(test_end) + 1
    comparisons = {}
    for label, source, target in (("bang_khae_local", "1040", "1040"),
                                  ("bang_kapi_to_bang_khae", "1006", "1040")):
        source_info, target_info = data[source], data[target]
        # Freeze report history at the test boundary. Otherwise a test-day
        # report could alter features for later days in the same test season.
        target_info = {**target_info, "prior": target_info["prior"].copy()}
        target_info["prior"][n_test_start:n_test_end] = target_info["prior"][
            n_test_start]
        train_di, train_ci, weights = training_indices(source_info["y"], n_train)
        test_di = np.repeat(np.arange(n_test_start, n_test_end), target_info["cells"])
        test_ci = np.tile(np.arange(target_info["cells"]), n_test_end - n_test_start)
        truth = target_info["y"][n_test_start:n_test_end].astype(np.uint8)
        target_train = source_info["y"][train_di, train_ci]
        comparisons[label] = {}
        for variant in VARIANTS:
            x_train = feature_matrix(source_info, train_di, train_ci, variant)
            x_test = feature_matrix(target_info, test_di, test_ci, variant)
            if x_train.shape[1] != x_test.shape[1]:
                raise ValueError("Source and target features differ")
            fitted = model().fit(x_train, target_train, sample_weight=weights)
            probability = fitted.predict_proba(x_test)[:, 1].reshape(truth.shape)
            comparisons[label][variant] = {
                "cell_rank": rank_metrics(probability.sum(axis=0), truth.sum(axis=0)),
                "daily_average_precision": round(float(average_precision_score(
                    truth.ravel(), probability.ravel())), 5),
                "daily_brier": round(float(brier_score_loss(
                    truth.ravel(), probability.ravel())), 6),
                "predicted_positive_cell_days": round(float(probability.sum()), 1),
                "observed_positive_cell_days": int(truth.sum()),
                "train_positives": int(target_train.sum()),
            }
            print(label, variant, comparisons[label][variant], flush=True)
    return comparisons


def main() -> None:
    levels, _ = river_levels()
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   CACHE / "rain_1006_2021_2025.json"))
    river = river_features(days, levels)
    stations, locations, gauge_rain = rain_tables(days)
    data, rain_audit = {}, {}
    feature_names = {}
    for code in CODES:
        info = load_district(code, days, river)
        with np.load(OUTPUT / f"surface_shared_state_{code}_daily_cells.npz") as saved:
            rr, cc = saved["rows"], saved["cols"]
        info["new_static"], feature_names["new_static"] = load_group(
            code, "new_static", rr, cc)
        info["canal_graph"], feature_names["canal_graph"] = load_group(
            code, "canal_graph", rr, cc)
        info["spatial_rain"], rain_audit[code] = spatial_rain(
            code, days, stations, locations, gauge_rain, rr, cc)
        data[code] = info
    comparisons = evaluate_fold(data, days, TRAIN_END, date(2025, 10, 1),
                                date(2025, 12, 31))
    prior_season = evaluate_fold(data, days, date(2024, 7, 31),
                                 date(2024, 10, 1), date(2024, 12, 31))
    result = {
        "target": "100 m Traffy flood-report buffer touches a 250 m cell-day; not verified inundation",
        "train_dates": ["2022-01-01", str(TRAIN_END)],
        "test_dates": ["2025-10-01", "2025-12-31"],
        "prior_season_train_dates": ["2022-01-01", "2024-07-31"],
        "prior_season_test_dates": ["2024-10-01", "2024-12-31"],
        "training": "All positive cell-days plus identical 20% sampled negatives with inverse-probability weights; same tree settings",
        "baseline_features": BASE_FEATURES,
        "added_feature_names": feature_names,
        "hii_feature_names": RAIN_NAMES,
        "rain_interpolation": "Inverse-distance-squared with 2 km softening, stations within 20 km, only complete quality-screened station-days; missing remains NaN",
        "hii_timestamp_limit": "Published timestamps used as-is; time zone not independently verified",
        "rain_audit": rain_audit,
        "static_limit": "2021 WorldCover and 2025 OSM applied to earlier training years; D8 terrain flow and OSM topology are not engineered drainage",
        "test_limit": "Two exploratory October-December seasons in one target district; 2025 was repeatedly inspected before this run, and 2024 is a second check rather than independent external validation",
        "comparisons": comparisons,
        "prior_season_comparisons": prior_season,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
