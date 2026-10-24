"""Positive-only GISTDA check of a 250 m surface-flow scenario proxy.

The 10 m BMA DTM defines each cell's storage curve and candidate low crossing.
This is deliberately a coarse, closed-domain rain/overland-flow experiment: no
drains, pumps, infiltration, tide, buildings-as-walls, or observed water depth.
The 20 October 2026 polygon is held out from scenario selection.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box, shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/flood_bangkapi_surface/lat_krabang_edge_flow"
DTM = ROOT / "data/flood_sfincs_transfer/dtm_lat_krabang_10m.tif"
SURFACE = ROOT / "artifacts/flood_bangkapi_surface/sfincs_surface_multiarea/regions/lat_krabang/surface_10m.npz"
POLYGONS = ROOT / "artifacts/flood_bangkapi_surface"
SOURCE_DAYS = ("02", "03", "04", "07", "09", "10")
TEST_DAY = "20"
SCENARIOS = ((20, 2), (60, 2), (120, 6))  # millimetres, hours; controlled inputs
FRACTIONS = (.05, .10, .20)
GRID = 25  # 25 x 10 m = 250 m
CELL_M = 250
PIXEL_AREA = 100
CELL_AREA = CELL_M * CELL_M
WEIR_C = 1.6  # exploratory broad-crested coefficient; uncalibrated
STEP_S = 30  # 120 s produced a non-converged, oscillating flow solution


def grid_data():
    with rasterio.open(DTM) as ds:
        if ds.crs.to_epsg() != 32647 or ds.res != (10., 10.):
            raise ValueError("Unexpected DTM CRS or resolution")
        size = min(ds.height, ds.width) // GRID * GRID
        z = ds.read(1)[:size, :size].astype(np.float64)
        transform = ds.transform
    if not np.isfinite(z).all():
        raise ValueError("DTM contains invalid elevations")
    n = size // GRID
    blocks = z.reshape(n, GRID, n, GRID).transpose(0, 2, 1, 3)
    flat = blocks.reshape(n * n, GRID * GRID)
    sorted_z = np.sort(flat, axis=1)
    levels_v = (sorted_z * np.arange(1, GRID * GRID + 1)[None, :]
                - np.cumsum(sorted_z, axis=1)) * PIXEL_AREA
    prefix = np.cumsum(sorted_z, axis=1)
    east_pairs = np.maximum(blocks[:, :-1, :, -1], blocks[:, 1:, :, 0])
    south_pairs = np.maximum(blocks[:-1, :, -1, :], blocks[1:, :, 0, :])
    east_pass = east_pairs.min(axis=-1)
    south_pass = south_pairs.min(axis=-1)
    east_width = 10 * (east_pairs <= east_pass[..., None] + .10).sum(axis=-1)
    south_width = 10 * (south_pairs <= south_pass[..., None] + .10).sum(axis=-1)
    cell_ids = np.arange(n * n).reshape(n, n)
    src = np.r_[cell_ids[:, :-1].ravel(), cell_ids[:-1, :].ravel()]
    dst = np.r_[cell_ids[:, 1:].ravel(), cell_ids[1:, :].ravel()]
    saddle = np.r_[east_pass.ravel(), south_pass.ravel()]
    width = np.r_[east_width.ravel(), south_width.ravel()].astype(float)
    # Coarse control: only adjacent mean-height cells, with no subcell crossing.
    mean_z = flat.mean(axis=1)
    ordinary_pass = np.maximum(mean_z[src], mean_z[dst])
    with np.load(SURFACE) as surface:
        cover = surface["worldcover"][:size, :size]
    if cover.shape != z.shape:
        raise ValueError("Cover and DTM grid mismatch")
    crop = (cover.reshape(n, GRID, n, GRID).transpose(0, 2, 1, 3)
            .reshape(n * n, GRID * GRID) == 40).mean(axis=1)
    urban = (cover.reshape(n, GRID, n, GRID).transpose(0, 2, 1, 3)
             .reshape(n * n, GRID * GRID) == 50).mean(axis=1)
    return dict(z=z, n=n, transform=transform, cover=cover, flat=flat, sorted_z=sorted_z,
                levels_v=levels_v, prefix=prefix, src=src, dst=dst,
                saddle=saddle, width=width, ordinary_pass=ordinary_pass,
                crop=crop, urban=urban, mean_z=mean_z)


def surface_height(v: np.ndarray, data: dict) -> np.ndarray:
    """Invert exact 10 m subpixel storage within each 250 m cell."""
    zs, levels, prefix = data["sorted_z"], data["levels_v"], data["prefix"]
    eta = zs[:, 0].copy()
    for i in np.flatnonzero(v > 0):
        k = min(zs.shape[1], int(np.searchsorted(levels[i], v[i], side="right")))
        eta[i] = (v[i] / PIXEL_AREA + prefix[i, k - 1]) / k
    return eta


def simulate(data: dict, rain_mm: float, hours: float, arm: str):
    src, dst = data["src"], data["dst"]
    if arm == "ordinary":
        crest = data["ordinary_pass"]
        width = np.full(len(src), CELL_M, dtype=float)
    elif arm in ("saddle", "reversed"):
        crest = data["saddle"]
        width = data["width"]
    else:
        raise ValueError(arm)
    v = np.zeros(data["n"] ** 2, dtype=float)
    rain_v = rain_mm / 1000 * CELL_AREA
    steps = round(hours * 3600 / STEP_S)
    max_ledger_error = 0.0
    for step in range(steps):
        v += rain_v / steps
        eta = surface_height(v, data)
        high_is_src = eta[src] >= eta[dst]
        high = np.where(high_is_src, src, dst)
        low = np.where(high_is_src, dst, src)
        head = np.maximum(eta[high] - np.maximum(eta[low], crest), 0)
        transfer = WEIR_C * width * head ** 1.5 * STEP_S
        if arm == "reversed":
            high, low = low, high  # deliberate anti-gradient negative control
        requested = np.bincount(high, weights=transfer, minlength=len(v))
        factor = np.minimum(1, v / np.maximum(requested, 1e-12))
        transfer *= factor[high]
        np.add.at(v, high, -transfer)
        np.add.at(v, low, transfer)
        if v.min() < -1e-6:
            raise ValueError("Negative water storage")
        error = abs(v.sum() - (step + 1) * rain_v * len(v) / steps)
        max_ledger_error = max(max_ledger_error, error)
    eta = surface_height(v, data)
    pixel_depth = np.maximum(eta[:, None] - data["flat"], 0)
    mean_depth = v / CELL_AREA
    if not np.allclose(pixel_depth.mean(axis=1), mean_depth, atol=1e-7):
        raise ValueError("Storage curve inversion failed")
    return {"mean_depth_m": mean_depth,
            "wet_fraction_gt_10cm": (pixel_depth > .10).mean(axis=1),
            "wet_fraction_gt_2cm": (pixel_depth > .02).mean(axis=1),
            "ledger_error_m3": max_ledger_error,
            "total_rain_m3": rain_v * len(v),
            "max_mean_depth_m": float(mean_depth.max())}


def check_flow_direction() -> None:
    """A known two-cell slope must send water downhill and conserve volume."""
    flat = np.asarray([[1.] * 625, [0.] * 625, [0.] * 625, [0.] * 625])
    zs = np.sort(flat, axis=1)
    test = {"n": 2, "flat": flat, "sorted_z": zs,
            "levels_v": (zs * np.arange(1, 626)[None, :]
                         - np.cumsum(zs, axis=1)) * PIXEL_AREA,
            "prefix": np.cumsum(zs, axis=1),
            "src": np.asarray([0]), "dst": np.asarray([1]),
            "saddle": np.asarray([1.]), "ordinary_pass": np.asarray([1.]),
            "width": np.asarray([10.])}
    downhill = simulate(test, 20, .1, "saddle")["mean_depth_m"]
    reversed_flow = simulate(test, 20, .1, "reversed")["mean_depth_m"]
    assert downhill[1] > downhill[0]
    assert reversed_flow[0] > reversed_flow[1]
    assert np.isclose(downhill.sum(), .08) and np.isclose(reversed_flow.sum(), .08)


def mapped_geometry(data: dict, day: str):
    path = POLYGONS / f"gistda_daily_202610{day}_bma.geojson"
    features = json.loads(path.read_text())["features"]
    to_utm = Transformer.from_crs(4326, 32647, always_xy=True).transform
    terrain_box = box(data["transform"].c,
                      data["transform"].f - data["n"] * CELL_M,
                      data["transform"].c + data["n"] * CELL_M,
                      data["transform"].f)
    polygons = [shapely.transform(shape(f["geometry"]), to_utm,
                                  interleaved=False)
                for f in features if f["properties"].get("district_code") == "1011"]
    if not polygons:
        raise ValueError(f"No Lat Krabang positive polygons on {day}")
    return unary_union(polygons).intersection(terrain_box)


def label_area(data: dict, day: str) -> np.ndarray:
    mapped = mapped_geometry(data, day)
    left, top = data["transform"].c, data["transform"].f
    cells = np.asarray([box(left + col * CELL_M, top - (row + 1) * CELL_M,
                            left + (col + 1) * CELL_M, top - row * CELL_M)
                        for row in range(data["n"]) for col in range(data["n"])])
    areas = shapely.area(shapely.intersection(cells, mapped))
    if not np.isclose(areas.sum(), mapped.area, rtol=1e-5):
        raise ValueError(f"Mapped area does not sum on {day}")
    return areas


def capture(scores: np.ndarray, wet: np.ndarray, fraction: float) -> float:
    """Tie-aware expected mapped-area capture for a fixed cell budget."""
    budget = fraction * len(scores)
    weights = np.zeros(len(scores))
    used = 0.0
    for value in np.unique(scores)[::-1]:
        ids = np.flatnonzero(scores == value)
        share = min(1.0, max(0.0, (budget - used) / len(ids)))
        weights[ids] = share
        used += share * len(ids)
        if used >= budget - 1e-9:
            break
    return float(np.dot(weights, wet) / wet.sum())


def save_map(data: dict, wet: np.ndarray, scores: dict[str, np.ndarray],
             scenario: str) -> None:
    n = data["n"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), dpi=150,
                             constrained_layout=True)
    panels = [("GISTDA mapped water, 20 Oct", wet / CELL_AREA, "Blues"),
              ("Crop-cover reference", scores["crop_cover"], "YlGn"),
              ("Ordinary flow: share >10 cm", scores["ordinary"], "magma"),
              ("10 m saddle flow: share >10 cm", scores["saddle"], "magma")]
    for ax, (title, arr, cmap) in zip(axes, panels):
        im = ax.imshow(arr.reshape(n, n), cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=.76)
    fig.suptitle(f"Lat Krabang pilot: {scenario} controlled rain; 250 m cells", fontsize=12)
    fig.text(.5, .01, "Outside GISTDA polygons is unknown, not observed dry. Depth is simulated, not measured.",
             ha="center", fontsize=8)
    fig.savefig(OUT / "holdout_20261020_map.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check_flow_direction()
    data = grid_data()
    # Geometry/edge direction and finite-volume invariants, independent of labels.
    assert data["n"] == 34 and len(data["src"]) == 2 * data["n"] * (data["n"] - 1)
    assert np.all((data["width"] >= 10) & (data["width"] <= CELL_M))
    # Recompute a few boundary passes directly from the source raster.
    z = data["z"]
    for row, col in ((0, 0), (5, 17), (32, 32)):
        expected = min(max(z[row * GRID + k, (col + 1) * GRID - 1],
                           z[row * GRID + k, (col + 1) * GRID])
                       for k in range(GRID))
        assert np.isclose(data["saddle"][row * (data["n"] - 1) + col], expected)
    source_labels = {day: label_area(data, day) for day in SOURCE_DAYS}
    scores = {"crop_cover": data["crop"]}
    scenario_outputs = {}
    for mm, hours in SCENARIOS:
        scenario = f"{mm}mm_{hours}h"
        scenario_outputs[scenario] = {}
        for arm in ("ordinary", "saddle", "reversed"):
            scenario_outputs[scenario][arm] = simulate(data, mm, hours, arm)
        print("simulated", scenario, flush=True)
    source_crop_capture = round(float(np.mean([
        capture(data["crop"], wet, .10) for wet in source_labels.values()])), 5)
    source_results = {}
    for scenario, models in scenario_outputs.items():
        source_results[scenario] = {}
        for arm, output in models.items():
            score = output["wet_fraction_gt_10cm"]
            source_results[scenario][arm] = {
                "top_10pct_mapped_area_capture_mean": round(float(np.mean([
                    capture(score, wet, .10) for wet in source_labels.values()])), 5),
                "score": "fraction of 10 m subpixels over 10 cm simulated depth"}
    # This selection sees 2-10 October only. The 20 October polygons are read below.
    selected_scenario = max(source_results, key=lambda s: source_results[s]["saddle"][
        "top_10pct_mapped_area_capture_mean"])
    selected_mm, selected_hours = next((mm, hours) for mm, hours in SCENARIOS
                                    if selected_scenario == f"{mm}mm_{hours}h")
    # Numerics only: smaller time step, with identical terrain and scenario.
    global STEP_S
    reference_step = STEP_S
    STEP_S = 10
    fine = simulate(data, selected_mm, selected_hours, "saddle")
    STEP_S = reference_step
    coarse = scenario_outputs[selected_scenario]["saddle"]
    numerical_audit = {
        "step_s": reference_step, "reference_step_s": 10,
        "mean_depth_mae_mm": round(float(np.mean(np.abs(
            coarse["mean_depth_m"] - fine["mean_depth_m"]))) * 1000, 4),
        "max_depth_difference_mm": round(abs(coarse["max_mean_depth_m"]
                                           - fine["max_mean_depth_m"]) * 1000, 4)}
    holdout = label_area(data, TEST_DAY)
    # Raster category overlay is approximate at 10 m; it diagnoses the sensor
    # target mix and is never used in source scenario selection.
    mapped_pixels = rasterize([(mapped_geometry(data, TEST_DAY), 1)],
                              out_shape=data["z"].shape, transform=data["transform"],
                              fill=0).astype(bool)
    cover_breakdown = {str(int(code)): round(float(np.count_nonzero(
        mapped_pixels & (data["cover"] == code)) / mapped_pixels.sum()), 4)
        for code in np.unique(data["cover"])}
    holdout_scores = {"crop_cover": data["crop"]}
    for arm in ("ordinary", "saddle", "reversed"):
        holdout_scores[arm] = scenario_outputs[selected_scenario][arm]["wet_fraction_gt_10cm"]
    results = {}
    for name, score in holdout_scores.items():
        results[name] = {f"top_{round(f*100)}pct_mapped_area_capture": round(
            capture(score, holdout, f), 5) for f in FRACTIONS}
        results[name]["distinct_scores"] = int(len(np.unique(score)))
    holdout_scenarios = {
        scenario: {arm: round(capture(output["wet_fraction_gt_10cm"], holdout, .10), 5)
                   for arm, output in models.items()}
        for scenario, models in scenario_outputs.items()}
    for scenario, models in scenario_outputs.items():
        for arm, output in models.items():
            np.save(OUT / f"{scenario}_{arm}_mean_depth_m.npy",
                    output["mean_depth_m"].reshape(data["n"], data["n"]).astype("float32"))
    transform = Affine(data["transform"].a * GRID, 0, data["transform"].c,
                       0, data["transform"].e * GRID, data["transform"].f)
    with rasterio.open(OUT / "holdout_selected_saddle_depth_proxy_250m.tif", "w",
                       driver="GTiff", height=data["n"], width=data["n"], count=1,
                       dtype="float32", crs="EPSG:32647", transform=transform) as dst:
        dst.write(scenario_outputs[selected_scenario]["saddle"]["mean_depth_m"].reshape(
            data["n"], data["n"]).astype("float32"), 1)
    np.savez_compressed(OUT / "holdout_20261020_scores_labels.npz",
                        gistda_positive_area_m2=holdout.reshape(data["n"], data["n"]),
                        crop_cover=data["crop"].reshape(data["n"], data["n"]),
                        built_cover=data["urban"].reshape(data["n"], data["n"]),
                        **{name: value.reshape(data["n"], data["n"])
                           for name, value in holdout_scores.items() if name != "crop_cover"})
    save_map(data, holdout, holdout_scores, selected_scenario)
    summary = {
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "area": "Lat Krabang DTM window, 8.5 by 8.5 km; 34 by 34 250 m cells",
        "source_dates_for_scenario_selection": [f"2026-10-{x}" for x in SOURCE_DAYS],
        "held_out_date": "2026-10-20", "selected_scenario": selected_scenario,
        "scenarios": [f"{mm}mm_{hours}h" for mm, hours in SCENARIOS],
        "source_selection": source_results,
        "source_crop_top_10pct_mapped_area_capture_mean": source_crop_capture,
        "held_out_positive_area_km2": round(float(holdout.sum()) / 1e6, 5),
        "held_out_positive_cells": int((holdout > 1).sum()),
        "held_out_mapped_10m_landcover_share": cover_breakdown,
        "held_out_capture": results,
        "held_out_top_10pct_scenario_sensitivity": holdout_scenarios,
        "solver_audit": {scenario: {arm: {key: round(float(output[key]), 6)
                                           for key in ("ledger_error_m3", "total_rain_m3",
                                                       "max_mean_depth_m")}
                                    for arm, output in models.items()}
                         for scenario, models in scenario_outputs.items()},
        "numerical_audit": numerical_audit,
        "edge_audit": {"edges": int(len(data["src"])),
                       "min_saddle_width_m": float(data["width"].min()),
                       "median_saddle_width_m": float(np.median(data["width"])),
                       "median_saddle_minus_ordinary_pass_m": round(float(np.median(
                           data["saddle"] - data["ordinary_pass"])), 4)},
        "inputs": {"dtm": str(DTM.relative_to(ROOT)),
                   "worldcover": str(SURFACE.relative_to(ROOT)),
                   "GISTDA": "daily BMA-clipped polygons, district 1011"},
        "interpretation": "Capture of GISTDA-positive mapped area at fixed cell budget. Unmapped is unknown; all depth values are modeled scenario proxies.",
        "limitations": ["GISTDA acquisition time and complete observable footprint are unverified",
                        "rain is controlled, not date-matched; scenarios are not a hindcast",
                        "10 m boundary minimum is a candidate opening, not a surveyed passage",
                        "instant mixing within 250 m storage cells; no within-cell routing",
                        "closed domain without drainage, pumping, infiltration, tide or barriers",
                        "source dates and holdout may share spatial flood patterns",
                        "crop cover is a surface category, not measured absorption"]}
    (OUT / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"selected_scenario": selected_scenario,
                      "source_selection": source_results,
                      "holdout": results}, indent=2))


if __name__ == "__main__":
    main()
