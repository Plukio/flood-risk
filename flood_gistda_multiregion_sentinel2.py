"""Build pre-event Sentinel-2 vegetation layers for six 250 m flood patches.

The 2025 dry-season composite predates every 2026 GISTDA label. SCL masks
cloud/shadow; unobserved pixels do not become bare land.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import numpy as np
import planetary_computer
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from flood_gistda_multiregion_data import CACHE, OUTPUT, GRID_M, SIZE, specs


STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
WINDOW = "2025-01-01T00:00:00Z/2025-04-30T23:59:59Z"
SUBPIXEL_M = 50
SUBSIZE = SIZE * GRID_M // SUBPIXEL_M
CHANNELS = ("s2_pre_event_ndvi", "s2_pre_event_green_fraction",
            "s2_pre_event_valid_fraction")


def search(spec: dict) -> list[dict]:
    response = requests.post(STAC, json={
        "collections": ["sentinel-2-l2a"], "bbox": spec["query_bbox"],
        "datetime": WINDOW, "query": {"eo:cloud_cover": {"lt": 20}},
        "limit": 100}, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if any(link["rel"] == "next" for link in payload.get("links", [])):
        raise ValueError("STAC scene list is paginated; increase search handling")
    groups: dict[str, list] = {}
    for item in payload["features"]:
        tile = item["id"].split("_")[4]
        groups.setdefault(tile, []).append(item)
    # Two low-cloud acquisitions per intersecting MGRS tile. A patch may cross
    # a tile boundary, so one item need not cover its entire extent.
    chosen = []
    for tile, items in groups.items():
        chosen.extend(sorted(items, key=lambda item: (
            item["properties"]["eo:cloud_cover"], item["id"]))[:2])
    if not chosen:
        raise ValueError("No pre-event Sentinel-2 scenes for patch")
    return chosen


def read_band(item: dict, name: str, transform) -> np.ndarray:
    href = planetary_computer.sign(item["assets"][name]["href"])
    resampling = Resampling.nearest if name == "SCL" else Resampling.bilinear
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
        with rasterio.open(href) as source:
            with WarpedVRT(source, crs="EPSG:32647", transform=transform,
                           width=SUBSIZE, height=SUBSIZE,
                           resampling=resampling) as warped:
                return warped.read(1)


def composite(name: str, spec: dict) -> dict:
    scenes = search(spec)
    west, _, _, north = spec["utm_bounds"]
    transform = from_origin(west, north, SUBPIXEL_M, SUBPIXEL_M)
    values = []
    for item in scenes:
        red = read_band(item, "B04", transform).astype(np.float32)
        nir = read_band(item, "B08", transform).astype(np.float32)
        scl = read_band(item, "SCL", transform)
        good = np.isin(scl, (4, 5, 6)) & (red > 0) & (nir > 0)
        ndvi = np.full(red.shape, np.nan, dtype=np.float32)
        ndvi[good] = (nir[good] - red[good]) / (nir[good] + red[good])
        values.append(ndvi)
        print("S2", name, item["id"], "valid", round(float(good.mean()), 3), flush=True)
    cube = np.stack(values)
    # nanmedian warns for completely unobserved pixels, handled explicitly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(cube, axis=0)
    valid = np.isfinite(median)
    per_cell = median.reshape(SIZE, 5, SIZE, 5).transpose(0, 2, 1, 3)
    mask = valid.reshape(SIZE, 5, SIZE, 5).transpose(0, 2, 1, 3)
    fraction = mask.mean(axis=(2, 3)).astype(np.float32)
    ndvi_sum = np.nansum(per_cell, axis=(2, 3))
    ndvi_cell = np.divide(ndvi_sum, mask.sum(axis=(2, 3)),
                          out=np.zeros((SIZE, SIZE), np.float32),
                          where=mask.sum(axis=(2, 3)) > 0).astype(np.float32)
    green = np.divide(((per_cell > .3) & mask).sum(axis=(2, 3)),
                      mask.sum(axis=(2, 3)),
                      out=np.zeros((SIZE, SIZE), np.float32),
                      where=mask.sum(axis=(2, 3)) > 0).astype(np.float32)
    layer = np.stack((ndvi_cell, green, fraction))
    np.savez_compressed(CACHE / f"{name}_sentinel2_2025_250m.npz",
                        values=layer, names=np.asarray(CHANNELS))
    return {"scene_ids": [item["id"] for item in scenes],
            "scene_cloud_pct": [item["properties"]["eo:cloud_cover"] for item in scenes],
            "cells_at_least_80pct_valid": int(np.count_nonzero(fraction >= .8)),
            "mean_valid_fraction": round(float(fraction.mean()), 4),
            "mean_ndvi_observed_cells": round(float(ndvi_cell[fraction > 0].mean()), 4)}


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous_path = OUTPUT / "sentinel2_2025_audit.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    previous_patches = previous.get("patches", {})
    audit = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source": STAC, "collection": "sentinel-2-l2a",
             "pre_event_window": WINDOW, "cloud_threshold_scene_pct": 20,
             "scl_accepted_classes": [4, 5, 6], "subpixel_m": SUBPIXEL_M,
             "green_ndvi_threshold": .3, "channels": CHANNELS,
             "meaning": "Pre-event vegetation proxy, not measured soil infiltration or verified permeable area",
             "patches": {}}
    for name, spec in specs().items():
        if name in previous_patches and (CACHE / f"{name}_sentinel2_2025_250m.npz").exists():
            audit["patches"][name] = previous_patches[name]
            print("S2 cached", name, flush=True)
        else:
            audit["patches"][name] = composite(name, spec)
    (OUTPUT / "sentinel2_2025_audit.json").write_text(json.dumps(
        audit, indent=2) + "\n")
    print("saved", OUTPUT / "sentinel2_2025_audit.json", flush=True)


if __name__ == "__main__":
    main()
