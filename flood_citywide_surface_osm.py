"""Rasterize OSM building footprints onto the existing 250 m Bangkok grid."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from flood_bangkapi_surface_grid import OSM_PBF, SurfaceOSM
from flood_citywide_history_ranker import SHAPE
from flood_multi_target_citywide_data import OUT, X0, Y0

OUTPUT = OUT / "osm_2025_building_footprint_citywide_250m.npz"
AUDIT = OUT / "osm_2025_building_footprint_citywide_250m_audit.json"


def build():
    if OUTPUT.exists() and AUDIT.exists():
        with np.load(OUTPUT) as saved:
            value = saved["building_fraction"]
        if value.shape != SHAPE or not np.isfinite(value).all():
            raise ValueError("Cached citywide buildings are invalid")
        return value
    rows, cols = SHAPE
    grid = {"west": X0, "south": Y0, "east": X0+cols*250,
            "north": Y0+rows*250, "rows": rows, "cols": cols}
    handler = SurfaceOSM(grid)
    handler.apply_file(str(OSM_PBF), locations=True)
    fraction = np.clip(handler.building_area_m2/(250.**2), 0, 1).astype(np.float32)
    if fraction.shape != SHAPE or not np.isfinite(fraction).all():
        raise ValueError("OSM building fraction invalid")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, building_fraction=fraction)
    AUDIT.write_text(json.dumps({
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(OSM_PBF), "osm_snapshot": "2025-01-01",
        "method": "Closed OSM building ways clipped to 250 m UTM47N cells; footprint area/cell area, clipped to 1",
        "features_seen_in_bbox": dict(handler.features_seen),
        "nonzero_building_cells": int(np.count_nonzero(fraction)),
        "mean_fraction_where_nonzero": float(fraction[fraction>0].mean())
        if np.any(fraction>0) else 0.}, indent=2)+"\n")
    return fraction


if __name__ == "__main__":
    print("building cells", int(np.count_nonzero(build())), flush=True)
