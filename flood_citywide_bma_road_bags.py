"""Locate BMA road-flood reports and comparison roads as OSM road-cell bags.

The bag is every 250 m candidate cell touched by a matching named OSM road
inside the reported district. It is not an exact incident coordinate.

Run: .venv/bin/python flood_citywide_bma_road_bags.py
"""
from __future__ import annotations

import json
import zlib
from collections import Counter, defaultdict
from datetime import date, datetime, timezone

import numpy as np
import osmium
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString

from flood_bangkok_multitask_depth_cnn import CACHE, csv_rows, normalize_name
from flood_bangkapi_surface_grid import OSM_PBF, TO_UTM
from flood_citywide_history_ranker import SHAPE, load_grid
from flood_multi_target_citywide_data import OUT, X0, Y0, ROWS, COLS, TILE_M
from flood_multi_target_data import START

CACHE_NPZ = OUT / "bma_road_bags_2022_2024.npz"
AUDIT_JSON = OUT / "bma_road_bags_2022_2024_audit.json"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def road_name(value: str) -> str:
    name = normalize_name(value).translate(THAI_DIGITS)
    for prefix in ("ถนน", "ซอย"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def bma_rows(data: dict) -> list[dict]:
    district_by_th = {d["name_th"]: i for i, d in enumerate(data["districts"])}
    rows = []
    for year in (2022, 2023, 2024):
        for record in csv_rows(CACHE / f"bma_flood_{year}.csv"):
            if not record["frd_flood_height"].strip():
                continue
            district = district_by_th.get(
                record["district"].replace("เขต", "").strip())
            name = road_name(record["frd_road"])
            if district is None or not name:
                continue
            day = datetime.strptime(record["frd_flood_date_begin"],
                                    "%m/%d/%Y").date()
            if day.year != year:
                continue
            rows.append({"day": (day-START).days, "year": year,
                         "district": district, "road": name,
                         "depth_cm": float(record["frd_flood_height"]),
                         "area_text": record["frd_area"]})
    return rows


class NamedRoads(osmium.SimpleHandler):
    def __init__(self, wanted: set[str]):
        super().__init__()
        self.wanted = wanted
        self.lines = defaultdict(list)
        self.matched_ways = 0

    def way(self, way):
        if not way.tags.get("highway"):
            return
        names = {road_name(way.tags.get(key, ""))
                 for key in ("name", "name:th")}
        names = {name for name in names if name and (
            name in self.wanted or (len(name) >= 4 and
            zlib.crc32(name.encode("utf-8")) % 4 == 0))}
        if not names:
            return
        points = [(node.lon, node.lat) for node in way.nodes
                  if node.location.valid()]
        if len(points) < 2:
            return
        lon = np.asarray([point[0] for point in points])
        lat = np.asarray([point[1] for point in points])
        if (lon.max() < 100.1 or lon.min() > 101.1 or
            lat.max() < 13.3 or lat.min() > 14.4):
            return
        x, y = TO_UTM.transform(lon, lat)
        line = LineString(zip(x, y))
        west, south, east, north = line.bounds
        if (east < X0 or west > X0+COLS*TILE_M or
            north < Y0 or south > Y0+ROWS*TILE_M):
            return
        for name in names:
            self.lines[name].append(line)
        self.matched_ways += 1


def build(data: dict):
    rows = bma_rows(data)
    raw_depth_rows = sum(bool(r["frd_flood_height"].strip())
                         for year in (2022, 2023, 2024)
                         for r in csv_rows(CACHE / f"bma_flood_{year}.csv"))
    targets = {(r["district"], r["road"]) for r in rows}
    roads = NamedRoads({name for _, name in targets})
    roads.apply_file(str(OSM_PBF), locations=True)
    transform = from_origin(X0, Y0+ROWS*TILE_M, 250, 250)
    district = data["district_id"]
    candidate = data["candidate"]
    flat = data["candidate_flat"]
    bags = []
    pairs = []
    for district_id, name in ((d, n) for d in range(len(data["districts"]))
                              for n in sorted(roads.lines)):
        polygon = data["districts"][district_id]["polygon"]
        lines = [line for line in roads.lines.get(name, [])
                 if line.intersects(polygon)]
        if not lines:
            continue
        road_mask = rasterize(
            [(line, 1) for line in lines], out_shape=SHAPE,
            transform=transform, fill=0, all_touched=True,
            dtype=np.uint8).astype(bool)
        chosen = np.flatnonzero((road_mask & candidate &
                                 (district == district_id)).ravel())
        if not len(chosen):
            continue
        indices = np.searchsorted(flat, chosen).astype(np.int32)
        if not np.array_equal(flat[indices], chosen):
            raise ValueError("Road bag has noncandidate cells")
        bags.append(indices)
        pairs.append((district_id, name))
    lookup = {pair: j for j, pair in enumerate(pairs)}
    aligned = [dict(r, bag=lookup[(r["district"], r["road"])])
               for r in rows if (r["district"], r["road"]) in lookup]
    reported = np.asarray([pair in targets for pair in pairs], bool)
    sizes = np.asarray([len(bag) for bag in bags], np.int32)
    records_by_bag = Counter(r["bag"] for r in aligned)
    audit = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_depth_rows": raw_depth_rows,
        "eligible_depth_rows": len(rows),
        "road_district_pairs": len(targets),
        "osm_named_ways_in_bangkok_extent": roads.matched_ways,
        "district_road_bags": len(pairs),
        "district_matched_reported_pairs": int(reported.sum()),
        "district_matched_rows": len(aligned),
        "sampled_unreported_road_bags": int((~reported).sum()),
        "sampled_unreported_road_rule": "CRC32(normalized OSM name) mod 4 = 0; no BMA depth report on that district-road pair in 2022-2024",
        "bag_size_quantiles_250m_cells": {
            str(q): float(np.percentile(sizes, q))
            for q in (0, 10, 25, 50, 75, 90, 95, 100)},
        "reported_bag_size_quantiles_250m_cells": {
            str(q): float(np.percentile(sizes[reported], q))
            for q in (0, 10, 25, 50, 75, 90, 95, 100)},
        "rows_by_bag_size_limit": {str(limit): sum(
            2 <= len(bags[r["bag"]]) <= limit for r in aligned)
            for limit in (10, 20, 30, 40, 60, 80)},
        "rows_by_year": dict(Counter(str(r["year"]) for r in aligned)),
        "max_records_for_one_road": max(records_by_bag.values()),
        "location_semantics": "All 250m candidate cells touched by a matching OSM road within the BMA district; area text is not yet geocoded",
        "sources": "BMA flood CSV 2022-2024, Thailand OSM 2025-01-01, Bangkok 250m candidate grid",
    }
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int32)
    CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_NPZ, candidate_flat=flat, offsets=offsets,
        bag_indices=np.concatenate(bags).astype(np.int32),
        bag_district=np.asarray([x[0] for x in pairs], np.int16),
        bag_road=np.asarray([x[1] for x in pairs]),
        bag_reported=reported,
        record_day=np.asarray([r["day"] for r in aligned], np.int16),
        record_year=np.asarray([r["year"] for r in aligned], np.int16),
        record_bag=np.asarray([r["bag"] for r in aligned], np.int16),
        record_depth=np.asarray([r["depth_cm"] for r in aligned], np.float32),
    )
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n")
    return audit


def load_bags(data: dict):
    if not CACHE_NPZ.exists():
        build(data)
    with np.load(CACHE_NPZ, allow_pickle=False) as z:
        if not np.array_equal(z["candidate_flat"], data["candidate_flat"]):
            raise ValueError("BMA road bags differ from current candidate grid")
        offsets = z["offsets"]
        bags = [z["bag_indices"][offsets[i]:offsets[i+1]]
                for i in range(len(offsets)-1)]
        return bags, z["bag_district"].copy(), z["record_day"].copy(), \
            z["record_bag"].copy(), z["record_depth"].copy(), \
            z["bag_reported"].copy()


if __name__ == "__main__":
    print(json.dumps(build(load_grid()), indent=2, ensure_ascii=False))
