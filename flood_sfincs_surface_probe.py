"""Check that mapped 10 m surface cover changes the pinned SFINCS teacher.

The roughness values and 10 m road corridor are scenario assumptions, not
measured hydraulic coefficients. This probe does not train a flood model.

Run ``.venv/bin/python flood_sfincs_surface_probe.py prepare``, run the
generated ``run_all.sh`` inside the pinned SFINCS container, then run
``.venv/bin/python flood_sfincs_surface_probe.py analyze``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import osmium
import planetary_computer
import rasterio
import requests
from netCDF4 import Dataset
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from shapely.geometry import LineString

from flood_bangkapi_surface_grid import OSM_PBF, ROAD_CLASSES
from flood_sfincs_lidar_trial import DTM, IMAGE
from flood_sfincs_physics_pretrain import OUT as OLD, SIZE


OUT = Path("artifacts/flood_bangkapi_surface/sfincs_surface_probe")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
# WorldCover code to Manning n; deliberately exploratory, not measured values.
ROUGHNESS = {10: .070, 20: .055, 30: .050, 40: .045, 50: .040,
             60: .030, 80: .025, 90: .055, 95: .055, 100: .060}
ROAD_N = .020
TILES = ((4, 3), (4, 8))


class RoadLines(osmium.SimpleHandler):
    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox
        self.lines = []
        self.to_utm = Transformer.from_crs(4326, 32647, always_xy=True)

    def way(self, way):
        if way.tags.get("highway") not in ROAD_CLASSES:
            return
        points = [(n.lon, n.lat) for n in way.nodes if n.location.valid()]
        if len(points) < 2:
            return
        west, south, east, north = self.bbox
        if (max(p[0] for p in points) < west or min(p[0] for p in points) > east
                or max(p[1] for p in points) < south
                or min(p[1] for p in points) > north):
            return
        xs, ys = self.to_utm.transform(*zip(*points))
        line = LineString(zip(xs, ys))
        if line.length:
            self.lines.append(line)


def static_layers(dtm_path: Path = DTM):
    with rasterio.open(dtm_path) as dtm:
        bbox = Transformer.from_crs(dtm.crs, 4326, always_xy=True).transform_bounds(
            *dtm.bounds)
        response = requests.post(STAC, json={
            "collections": ["esa-worldcover"], "bbox": bbox,
            "datetime": "2021-01-01T00:00:00Z/2021-12-31T23:59:59Z",
            "limit": 20}, timeout=30)
        response.raise_for_status()
        items = response.json()["features"]
        if len(items) != 1:
            raise ValueError(f"Expected one WorldCover tile, got {len(items)}")
        href = planetary_computer.sign(items[0]["assets"]["map"]["href"])
        with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
            with rasterio.open(href) as source:
                with WarpedVRT(source, crs=dtm.crs, transform=dtm.transform,
                               width=dtm.width, height=dtm.height,
                               resampling=Resampling.nearest) as warped:
                    cover = warped.read(1)
        if np.any(cover == 0) or set(np.unique(cover)) - set(ROUGHNESS):
            raise ValueError("Missing or unfamiliar land-cover classes")
        roads = RoadLines(bbox)
        roads.apply_file(str(OSM_PBF), locations=True)
        # A touched cell represents a mapped road corridor at 10 m resolution;
        # road width and paved fraction within the cell are not known.
        road = rasterize(((line, 1) for line in roads.lines),
                         out_shape=cover.shape, transform=dtm.transform,
                         fill=0, dtype="uint8", all_touched=True)
        n = np.full(cover.shape, .035, np.float32)
        for code, value in ROUGHNESS.items():
            n[cover == code] = value
        n[road != 0] = ROAD_N
        meta = {"worldcover_tile": items[0]["id"], "worldcover_year": 2021,
                "osm_pbf": str(OSM_PBF), "osm_snapshot": "2025-01-01",
                "grid": {"crs": str(dtm.crs), "transform": tuple(dtm.transform),
                         "shape": cover.shape, "cell_m": 10},
                "land_cover_counts": {str(int(k)): int(v) for k, v in
                                      zip(*np.unique(cover, return_counts=True))},
                "osm_road_lines_near_grid": len(roads.lines),
                "road_touched_cells": int(np.count_nonzero(road)),
                "manning_n_by_worldcover_class": ROUGHNESS,
                "manning_n_road_override": ROAD_N,
                "caveat": "Roughness and 10 m road corridor are scenario assumptions; "
                          "buildings, infiltration, pipes and pumps are absent."}
    return cover, road, n, meta


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "surface_10m.npz").exists() and (OUT / "surface_audit.json").exists():
        with np.load(OUT / "surface_10m.npz") as saved:
            n = saved["assumed_manning_n"]
        meta = json.loads((OUT / "surface_audit.json").read_text())
    else:
        cover, road, n, meta = static_layers()
        np.savez_compressed(OUT / "surface_10m.npz", worldcover=cover,
                            road_corridor=road, assumed_manning_n=n)
        (OUT / "surface_audit.json").write_text(json.dumps(meta, indent=2)+"\n")
    names = []
    for row, col in TILES:
        original = OLD / "cases" / f"r{row}c{col}_40mmh_15min"
        for arm in ("uniform_file", "mapped_roughness"):
            name = f"r{row}c{col}_{arm}"
            names.append(name)
            dest = OUT / "cases" / name
            dest.mkdir(parents=True, exist_ok=True)
            for fname in ("sfincs.dep", "sfincs.msk", "sfincs.prcp"):
                shutil.copyfile(original / fname, dest / fname)
            section = np.full((SIZE, SIZE), .035, np.float32) if arm == "uniform_file" else (
                n[row*SIZE:(row+1)*SIZE, col*SIZE:(col+1)*SIZE])
            if section.shape != (SIZE, SIZE):
                raise ValueError("Surface and DTM tile grids differ")
            # Manning binary is Fortran x-fastest, unlike NumPy row-major.
            # Transpose before writing; SFINCS's map grid then agrees with
            # the north-first terrain/WorldCover/road grid after a row flip.
            np.flipud(section).T.astype("<f4").tofile(dest / "sfincs.man")
            inp = (original / "sfincs.inp").read_text()
            inp = inp.replace("manning = 0.035\n", "manningfile = sfincs.man\n")
            (dest / "sfincs.inp").write_text(inp)
    (OUT / "case_names.txt").write_text("\n".join(names)+"\n")
    (OUT / "run_all.sh").write_text("""#!/bin/sh
set -eu
while IFS= read -r name; do
  cd "/data/cases/$name"
  sfincs > sfincs.stdout 2>&1
  test -s sfincs_map.nc
  echo "Completed $name"
done < /data/case_names.txt
""")
    print(json.dumps({"cases": names, "audit": meta}, indent=2))


def analyze():
    results = {}
    for row, col in TILES:
        name = f"r{row}c{col}"
        original = OLD / "cases" / f"{name}_40mmh_15min" / "sfincs_map.nc"
        arms = {"old_uniform": original}
        for arm in ("uniform_file", "mapped_roughness"):
            arms[arm] = OUT / "cases" / f"{name}_{arm}" / "sfincs_map.nc"
        maps = {}
        for arm, path in arms.items():
            with Dataset(path) as ds:
                maps[arm] = np.asarray(ds.variables["h"][:], np.float32)[:, ::-1, :]
                if arm == "old_uniform":
                    continue
                saved_n = np.asarray(ds.variables["manning"][:], np.float32)[::-1]
                with np.load(OUT / "surface_10m.npz") as surface:
                    n = surface["assumed_manning_n"][row*SIZE:(row+1)*SIZE,
                                                     col*SIZE:(col+1)*SIZE]
                expected_n = (np.full_like(n, .035) if arm == "uniform_file" else n)
                if np.max(np.abs(saved_n-expected_n)) > 1e-6:
                    raise ValueError(f"Manning and terrain map misaligned in {name}/{arm}")
        base = maps["old_uniform"]
        results[name] = {}
        for arm in ("uniform_file", "mapped_roughness"):
            diff = maps[arm][-1] - base[-1]
            results[name][arm] = {
                "final_mae_vs_old_uniform_mm": float(np.mean(np.abs(diff))*1000),
                "final_p95_abs_change_mm": float(np.quantile(np.abs(diff), .95)*1000),
                "final_p99_abs_change_mm": float(np.quantile(np.abs(diff), .99)*1000),
                "final_max_abs_change_mm": float(np.max(np.abs(diff))*1000),
                "final_mean_depth_mm": float(np.mean(maps[arm][-1])*1000),
                "final_wet_gt20mm_cells": int(np.count_nonzero(maps[arm][-1]>.02)),
            }
    result = {"target": "Synthetic SFINCS surface response, no observed flood labels",
              "cases": results}
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "analyze"))
    action = parser.parse_args().action
    {"prepare": prepare, "analyze": analyze}[action]()
