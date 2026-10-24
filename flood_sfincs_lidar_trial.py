"""Prepare a matched Bang Kapi LiDAR surface-flow trial for SFINCS.

Run ``.venv/bin/python flood_sfincs_lidar_trial.py prepare`` and then run the
official SFINCS image in each printed case directory. No observed flood data
are used; these are controlled synthetic rainfall scenarios.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from flood_surface_local_inertial import LocalInertial2D


DTM = Path("data/flood_bangkapi_surface/dtm_bangkapi_10m.tif")
OUT = Path("artifacts/flood_bangkapi_surface/sfincs_trial")
SIZE = 96
RAIN_MM_H = (30, 60)
IMAGE = "deltares/sfincs-cpu:sfincs-v2.4.1"


def crop() -> tuple[np.ndarray, float, float, float, dict]:
    with rasterio.open(DTM) as src:
        row = (src.height - SIZE) // 2
        col = (src.width - SIZE) // 2
        window = Window(col, row, SIZE, SIZE)
        z = src.read(1, window=window).astype(np.float64)
        transform = src.window_transform(window)
        if src.crs.to_epsg() != 32647 or not np.isfinite(z).all():
            raise ValueError("Expected finite EPSG:32647 LiDAR crop")
        if abs(transform.a - 10) > 1e-6 or abs(transform.e + 10) > 1e-6:
            raise ValueError("Expected north-first 10 m square grid")
        meta = {"source": str(DTM), "source_window_row_col": [row, col],
                "size": SIZE, "epsg": 32647, "x0_m": transform.c,
                "y0_m": transform.f - SIZE * 10, "cell_m": 10,
                "z_min_m": float(z.min()), "z_max_m": float(z.max())}
        return z, transform.c, transform.f - SIZE * 10, 10., meta


def prepare() -> None:
    z, x0, y0, cell, meta = crop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "domain.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.save(OUT / "elevation_north_first_m.npy", z)
    for rain in RAIN_MM_H:
        case = OUT / f"rain_{rain}mmh_15min"
        case.mkdir(parents=True, exist_ok=True)
        # SFINCS ASCII regular grids start at the southwest cell; x varies
        # fastest. The source GeoTIFF starts at the northwest cell.
        np.savetxt(case / "sfincs.dep", np.flipud(z), fmt="%.8f")
        np.savetxt(case / "sfincs.msk", np.ones_like(z, dtype=np.int8), fmt="%d")
        (case / "sfincs.prcp").write_text(f"0 {rain}\n900 {rain}\n")
        config = f"""x0 = {x0:.3f}
y0 = {y0:.3f}
mmax = {SIZE}
nmax = {SIZE}
dx = {cell:.3f}
dy = {cell:.3f}
rotation = 0
epsg = 32647
tref = 20261023 000000
tstart = 20261023 000000
tstop = 20261023 001500
depfile = sfincs.dep
mskfile = sfincs.msk
precipfile = sfincs.prcp
inputformat = asc
outputformat = net
zsini = {float(z.min()) - 1:.8f}
advection = 0
alpha = 0.4
huthresh = 0.0001
theta = 1.0
viscosity = 0
manning = 0.035
qinf = 0
dtmax = 2.5
dtout = 900
dtmaxout = 900
storecumprcp = 1
"""
        (case / "sfincs.inp").write_text(config)
        print(f"Prepared {case}")
    print(f"Image: {IMAGE}")


def compare() -> None:
    import xarray as xr

    z, x0, y0, cell, meta = crop()
    solver = LocalInertial2D(z, cell, manning_n=.035)
    results = {"image": IMAGE, "domain": meta, "duration_s": 900,
               "boundary": "closed", "infiltration": "zero",
               "scenarios": {}}
    for rain in RAIN_MM_H:
        case = OUT / f"rain_{rain}mmh_15min"
        with xr.open_dataset(case / "sfincs_map.nc") as ds:
            if ds.sizes["time"] != 2 or ds.sizes["m"] != SIZE or ds.sizes["n"] != SIZE:
                raise ValueError("Unexpected SFINCS output grid or times")
            # SFINCS n increases northward; rasterio row increases southward.
            x = ds.x.values
            y = ds.y.values
            if (not np.allclose(x[0], x0 + (np.arange(SIZE)+.5)*cell, atol=.05)
                    or not np.allclose(y[:, 0], y0 + (np.arange(SIZE)+.5)*cell, atol=.05)):
                raise ValueError("SFINCS output coordinates do not align with LiDAR")
            z_sfincs = np.flipud(ds.zb.values)
            terrain_error = float(np.max(np.abs(z_sfincs-z)))
            if terrain_error > 1e-5:
                raise ValueError(f"SFINCS terrain mismatch: {terrain_error}")
            sfincs = np.flipud(ds.h.isel(time=-1).values.astype(np.float64))
            rain_grid = np.flipud(ds.cumprcp.isel(timemax=-1).values)
            if not np.isfinite(sfincs).all() or sfincs.min() < -1e-5:
                raise ValueError("SFINCS final depth missing or negative")
            if not np.allclose(rain_grid, rain/1000*.25, atol=1e-6):
                raise ValueError("SFINCS cumulative rain mismatch")
        sfincs = np.maximum(sfincs, 0)
        local = solver.run(solver.initial(), 900,
                           rain_mps=rain/1000/3600, max_dt_s=2.5).depth
        diff = sfincs - local
        expected = rain/1000*.25*SIZE*SIZE*cell*cell
        corr = float(np.corrcoef(sfincs.ravel(), local.ravel())[0, 1])
        summary = {
            "input_rain_mm": rain*.25,
            "expected_rain_volume_m3": expected,
            "sfincs_output_rain_mm": float(np.mean(rain_grid)*1000),
            "sfincs_volume_m3": float(sfincs.sum()*cell*cell),
            "local_volume_m3": float(local.sum()*cell*cell),
            "sfincs_volume_error_pct_of_rain": float((sfincs.sum()*cell*cell-expected)/expected*100),
            "local_volume_error_pct_of_rain": float((local.sum()*cell*cell-expected)/expected*100),
            "sfincs_p50_mm": float(np.median(sfincs)*1000),
            "local_p50_mm": float(np.median(local)*1000),
            "sfincs_p95_mm": float(np.quantile(sfincs, .95)*1000),
            "local_p95_mm": float(np.quantile(local, .95)*1000),
            "sfincs_max_mm": float(sfincs.max()*1000),
            "local_max_mm": float(local.max()*1000),
            "mean_absolute_cell_difference_mm": float(np.mean(np.abs(diff))*1000),
            "p95_absolute_cell_difference_mm": float(np.quantile(np.abs(diff), .95)*1000),
            "depth_pearson": corr,
            "terrain_max_absolute_difference_mm": terrain_error*1000,
        }
        results["scenarios"][str(rain)] = summary
        np.save(case / "sfincs_final_depth_north_first_m.npy", sfincs)
        np.save(case / "local_final_depth_north_first_m.npy", local)
        print(rain, json.dumps(summary, indent=2))
    (OUT / "comparison.json").write_text(json.dumps(results, indent=2)+"\n")


def plot() -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, rain in enumerate(RAIN_MM_H):
        case = OUT / f"rain_{rain}mmh_15min"
        sfincs = np.load(case / "sfincs_final_depth_north_first_m.npy")*1000
        local = np.load(case / "local_final_depth_north_first_m.npy")*1000
        limit = float(np.quantile(np.concatenate((sfincs.ravel(), local.ravel())), .995))
        for col, (array, title) in enumerate(((sfincs, "SFINCS"),
                                               (local, "Local inertial"))):
            im = axes[row, col].imshow(array, cmap="Blues", vmin=0, vmax=limit)
            axes[row, col].set_title(f"{rain} mm/h · {title}")
            fig.colorbar(im, ax=axes[row, col], label="Final depth (mm)", shrink=.72)
        delta = sfincs-local
        bound = float(np.quantile(np.abs(delta), .99))
        im = axes[row, 2].imshow(delta, cmap="RdBu_r", vmin=-bound, vmax=bound)
        axes[row, 2].set_title(f"{rain} mm/h · SFINCS − local")
        fig.colorbar(im, ax=axes[row, 2], label="Depth difference (mm)", shrink=.72)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Bang Kapi 10 m LiDAR · 15 min uniform rain · closed edges, no drains")
    path = OUT / "depth_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "compare", "plot"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "compare":
        compare()
    elif args.action == "plot":
        plot()
