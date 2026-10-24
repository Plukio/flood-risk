"""Build matching 250 m static inputs for a nonoverlapping Bangkok tile grid."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from flood_bangkok_transfer import load_districts
from flood_bangkapi_surface_grid import OSM_PBF
from flood_gistda_multiregion_data import (
    CACHE as STATIC_CACHE, TO_LONLAT, osm_arrays, terrain,
)
from flood_gistda_province_expand_data import worldcover
from flood_open_weak_data import LocalWays, ROOT, all_specs


OUT = ROOT / "artifacts" / "flood_open_weak_train" / "multi_target" / "citywide"
X0, Y0, TILE_M = 640000, 1490000, 20000
COLS, ROWS = 4, 3


def specs_and_masks():
    districts = load_districts()
    bma = unary_union([d["polygon"] for d in districts])
    trained = all_specs()
    seen = unary_union([box(*trained[name]["utm_bounds"])
                        for name in ("bang_kapi_core", "nong_chok")])
    specs, masks = {}, {}
    for row in range(ROWS):
        for col in range(COLS):
            west, south = X0 + col*TILE_M, Y0 + row*TILE_M
            east, north = west+TILE_M, south+TILE_M
            tile = box(west,south,east,north)
            area = bma.intersection(tile).area
            if area < 1_000_000:
                continue
            corners = [TO_LONLAT.transform(x,y) for x in (west,east)
                       for y in (south,north)]
            lon, lat = TO_LONLAT.transform(west+TILE_M/2,south+TILE_M/2)
            name = f"bma_r{row}_c{col}"
            specs[name] = {"center_lonlat": [lon,lat],
                           "utm_bounds": [west,south,east,north],
                           "query_bbox": [min(x for x,_ in corners)-.003,
                                          min(y for _,y in corners)-.003,
                                          max(x for x,_ in corners)+.003,
                                          max(y for _,y in corners)+.003]}
            masks[name] = {"bma_area_km2": round(area/1e6,4),
                           "outside_training_patches_km2": round(
                               bma.difference(seen).intersection(tile).area/1e6,4)}
    if len(specs) != 9:
        raise ValueError(f"Expected 9 BMA-covering tiles, got {len(specs)}")
    return specs,masks


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    specs,masks=specs_and_masks()
    pending={n:s for n,s in specs.items()
             if not (STATIC_CACHE/f"{n}_static_250m.npz").exists()}
    static_audit={}
    if pending:
        handler=LocalWays(pending)
        handler.apply_file(str(OSM_PBF),locations=True)
        if not handler.canal or not handler.road:
            raise ValueError("No OSM canal or road ways across Bangkok tiles")
        print("OSM",len(handler.canal),"canals",len(handler.road),"roads",flush=True)
        for name,spec in pending.items():
            west,south,east,north=spec["utm_bounds"]
            def nearby(lines):
                return [line for line in lines if
                        line.bounds[0]<=east+1000 and line.bounds[2]>=west-1000
                        and line.bounds[1]<=north+1000 and line.bounds[3]>=south-1000]
            canals,roads=nearby(handler.canal),nearby(handler.road)
            if not canals or not roads:
                raise ValueError(f"Missing OSM features in {name}")
            elev=terrain(spec)
            canal_len,canal_dist=osm_arrays(spec,canals)
            road_len,road_dist=osm_arrays(spec,roads)
            values=np.stack((elev[0],elev[1],elev[2],
                             np.log1p(canal_len),np.log1p(canal_dist),
                             np.log1p(road_len),np.log1p(road_dist))).astype(np.float32)
            if values.shape!=(7,80,80) or not np.isfinite(values).all():
                raise ValueError(f"Invalid static features in {name}")
            np.savez_compressed(STATIC_CACHE/f"{name}_static_250m.npz",values=values)
            static_audit[name]={"canal_ways":len(canals),"road_ways":len(roads)}
            print("static",name,static_audit[name],flush=True)
    cover_audit=worldcover(specs)
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tile_origin_utm47n":[X0,Y0],"tile_m":TILE_M,
            "tile_specs":specs,"areas":masks,
            "static_source":"SRTM elevation; 2025 OSM roads/canals; WorldCover 2021",
            "static_build":static_audit,"worldcover":cover_audit}
    (OUT/"data_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    build()
