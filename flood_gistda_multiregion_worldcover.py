"""Build pre-event ESA WorldCover 2021 class fractions for six flood patches.

WorldCover is static context: classes are neither observed permeability nor
evidence that a cell was dry on a GISTDA product date.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import planetary_computer
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from flood_gistda_multiregion_data import CACHE, GRID_M, OUTPUT, SIZE, specs


STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PIXEL_M = 10
SUBSIZE = SIZE * GRID_M // PIXEL_M
CLASSES = {
    "worldcover_built_fraction_2021": (50,),
    "worldcover_crop_fraction_2021": (40,),
    "worldcover_tree_grass_fraction_2021": (10, 20, 30),
    "worldcover_bare_fraction_2021": (60,),
    "worldcover_wetland_fraction_2021": (90, 95),
    "worldcover_water_fraction_2021": (80,),
}
CHANNELS = tuple(CLASSES)


def items() -> list[dict]:
    bounds = [patch["query_bbox"] for patch in specs().values()]
    bbox = [min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds)]
    response = requests.post(STAC, json={
        "collections": ["esa-worldcover"], "bbox": bbox,
        "datetime": "2021-01-01T00:00:00Z/2021-12-31T23:59:59Z",
        "limit": 100}, timeout=45)
    response.raise_for_status()
    result = response.json()
    if any(link["rel"] == "next" for link in result.get("links", [])):
        raise ValueError("WorldCover STAC result unexpectedly paginated")
    found = result["features"]
    if not found:
        raise ValueError("No 2021 WorldCover tiles found")
    return found


def class_fractions(land: np.ndarray) -> tuple[np.ndarray, float]:
    if land.shape != (SUBSIZE, SUBSIZE):
        raise ValueError(f"WorldCover warp shape changed: {land.shape}")
    valid = land != 0
    blocks = valid.reshape(SIZE, 25, SIZE, 25)
    denom = blocks.sum(axis=(1, 3))
    coverage = float(valid.mean())
    if coverage < .99 or np.any(denom == 0):
        raise ValueError(f"WorldCover patch coverage too low: {coverage:.4f}")
    fractions = []
    for codes in CLASSES.values():
        counts = np.isin(land, codes).reshape(SIZE, 25, SIZE, 25).sum(axis=(1, 3))
        fractions.append((counts / denom).astype(np.float32))
    return np.stack(fractions), coverage


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    catalog = items()
    audit = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source": STAC, "collection": "esa-worldcover", "year": 2021,
             "input_pixel_m": PIXEL_M, "output_cell_m": GRID_M,
             "class_codes": {key: list(value) for key, value in CLASSES.items()},
             "channels": list(CHANNELS), "patches": {}}
    for name, spec in specs().items():
        west, _, _, north = spec["utm_bounds"]
        transform = from_origin(west, north, PIXEL_M, PIXEL_M)
        land = np.zeros((SUBSIZE, SUBSIZE), dtype=np.uint8)
        used = []
        for item in catalog:
            bounds = item["bbox"]
            patch = spec["query_bbox"]
            if bounds[2] <= patch[0] or patch[2] <= bounds[0] or bounds[3] <= patch[1] or patch[3] <= bounds[1]:
                continue
            url = planetary_computer.sign(item["assets"]["map"]["href"])
            with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
                with rasterio.open(url) as source:
                    with WarpedVRT(source, crs="EPSG:32647", transform=transform,
                                   width=SUBSIZE, height=SUBSIZE,
                                   resampling=Resampling.nearest) as warped:
                        tile = warped.read(1)
            overlap = (land != 0) & (tile != 0)
            if np.any(overlap & (land != tile)):
                raise ValueError(f"WorldCover tiles disagree on {name}")
            land[tile != 0] = tile[tile != 0]
            used.append(item["id"])
        values, coverage = class_fractions(land)
        np.savez_compressed(CACHE / f"{name}_worldcover_2021_250m.npz",
                            values=values, names=np.asarray(CHANNELS))
        audit["patches"][name] = {
            "tile_ids": used, "valid_pixel_fraction": round(coverage, 5),
            "mean_class_fractions": {
                channel: round(float(values[i].mean()), 5)
                for i, channel in enumerate(CHANNELS)}}
        print(name, audit["patches"][name], flush=True)
    (OUTPUT / "worldcover_2021_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
