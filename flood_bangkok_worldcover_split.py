"""Extract separate WorldCover classes for Bang Kapi and Bang Khae report cells."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from flood_bangkok_new_static_features import reduce_10m, worldcover_url
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import GRID_M, OUTPUT, grid_spec
from flood_gistda_multiregion_worldcover import CHANNELS, CLASSES


CODES = ("1006", "1040")


def main() -> None:
    districts = {d["code"]: d for d in load_districts() if d["code"] in CODES}
    url, tile_id = worldcover_url()
    audit = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "worldcover_item": tile_id, "channels": CHANNELS, "districts": {}}
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
        with rasterio.open(url) as world:
            for code in CODES:
                grid = grid_spec(districts[code]["polygon"])
                with WarpedVRT(world, crs="EPSG:32647", transform=grid["transform"],
                               width=grid["width"], height=grid["height"],
                               resampling=Resampling.nearest) as warped:
                    land = warped.read(1)
                rows, cols = grid["rows"], grid["cols"]
                denom = reduce_10m((land != 0).astype(np.float32), rows, cols,
                                   lambda b: b.sum(axis=(1, 3)))
                values = []
                for classes in CLASSES.values():
                    count = reduce_10m(np.isin(land, classes).astype(np.float32),
                                       rows, cols, lambda b: b.sum(axis=(1, 3)))
                    values.append(np.divide(count, denom,
                        out=np.zeros((rows, cols), dtype=np.float32),
                        where=denom > 0))
                with np.load(OUTPUT / f"surface_shared_state_{code}_daily_cells.npz") as saved:
                    rr, cc = saved["rows"], saved["cols"]
                array = np.column_stack([layer[rr, cc] for layer in values]).astype(np.float32)
                if (not np.isfinite(array).all() or
                        np.max(np.abs(array.sum(axis=1) - 1)) > 1e-5):
                    raise ValueError(f"WorldCover classes incomplete in {code}")
                np.savez_compressed(CACHE / f"worldcover_split_{code}_250m.npz",
                                    values=array, names=np.asarray(CHANNELS),
                                    rows=rr, cols=cc)
                audit["districts"][code] = {
                    "cells": len(rr), "mean_class_fractions": {
                        name: round(float(array[:, i].mean()), 5)
                        for i, name in enumerate(CHANNELS)}}
                print(code, audit["districts"][code], flush=True)
    (CACHE / "worldcover_split_250m_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
