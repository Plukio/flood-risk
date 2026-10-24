"""Build dated, spatially grouped Traffy report-site proxies on the city grid.

Only direct flood-category coordinates are used. A site event is a complete-
linkage cluster of same-day points; its representative is a report medoid.
"New" means every point in that cluster is farther than the tested radius
from every report on an earlier day in the preceding 365 days. This is a
reported-site definition, not a surveyed flood extent or a dry-ground label.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree, distance

from flood_bangkapi_surface_grid import TO_UTM
from flood_bangkapi_surface_model import DIRECT_TYPES
from flood_citywide_history_ranker import SHAPE
from flood_multi_target_citywide_data import COLS, OUT, ROWS, TILE_M, X0, Y0
from flood_multi_target_data import START
from flood_open_weak_data import ROOT


RADII = (100, 250, 500)
SOURCES = (
    ROOT / "data/flood_bangkok_transfer/traffy_flood_2022_2024_49_districts.csv",
    ROOT / "data/flood_bangkapi_surface/traffy_flood_2022_2024_minimal.csv",
    ROOT / "data/flood_bangkapi_surface/traffy_flood_2025_minimal.csv",
)
DAY_END = date(2025, 12, 31)
NORTH = Y0 + ROWS * TILE_M
EAST = X0 + COLS * TILE_M
OUT_JSON = OUT / "report_physical_site_data_audit.json"


def load_points() -> dict[str, np.ndarray]:
    rows = []
    for path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8-sig", newline="") as file:
            for raw in csv.DictReader(file):
                if raw["type"] not in DIRECT_TYPES:
                    continue
                day = date.fromisoformat(raw["timestamp"][:10])
                if not START <= day <= DAY_END:
                    continue
                x, y = TO_UTM.transform(float(raw["lon"]), float(raw["lat"]))
                if not (X0 <= x < EAST and Y0 < y <= NORTH):
                    continue
                col = int((x-X0)//250)
                row = int((NORTH-y)//250)
                rows.append(((day-START).days, x, y, row, col,
                             raw["timestamp"]))
    rows.sort()
    if not rows:
        raise ValueError("No direct-category Traffy points in city grid")
    arr = np.asarray([row[:5] for row in rows], np.float64)
    return {"day": arr[:, 0].astype(np.int32),
            "xy": arr[:, 1:3].astype(np.float64),
            "row": arr[:, 3].astype(np.int16),
             "col": arr[:, 4].astype(np.int16),
             "time": np.asarray([row[5] for row in rows])}


def distance_to_prior_year(points: dict[str, np.ndarray]) -> np.ndarray:
    """Nearest earlier-day point; the scored day's reports never enter history."""
    days, xy = points["day"], points["xy"]
    nearest = np.full(len(days), np.inf, np.float32)
    for day in np.unique(days):
        first = int(np.searchsorted(days, day))
        end = int(np.searchsorted(days, day, side="right"))
        past = int(np.searchsorted(days, day-365))
        if first > past:
            nearest[first:end] = cKDTree(xy[past:first]).query(xy[first:end])[0]
    return nearest


def cluster_day(xy: np.ndarray, radius: int) -> list[np.ndarray]:
    if len(xy) == 1:
        return [np.array([0], np.int32)]
    # Complete linkage bounds the diameter, avoiding long single-link chains.
    groups = fcluster(linkage(xy, method="complete", metric="euclidean"),
                      t=radius, criterion="distance")
    return [np.flatnonzero(groups == group).astype(np.int32)
            for group in np.unique(groups)]


def events(points: dict[str, np.ndarray], nearest: np.ndarray,
           radius: int, candidate: np.ndarray, district_id: np.ndarray) -> list[dict]:
    days, xy = points["day"], points["xy"]
    result = []
    for day in np.unique(days):
        first = int(np.searchsorted(days, day))
        end = int(np.searchsorted(days, day, side="right"))
        for local in cluster_day(xy[first:end], radius):
            member = local + first
            here = xy[member]
            medoid = member[np.argmin(distance.cdist(here, here).sum(axis=1))]
            row, col = int(points["row"][medoid]), int(points["col"][medoid])
            if not (0 <= row < SHAPE[0] and 0 <= col < SHAPE[1]
                    and candidate[row, col]):
                continue
            result.append({"day_index": int(day), "day": str(START.fromordinal(
                              START.toordinal()+int(day))),
                           "district_index": int(district_id[row, col]),
                           "row": row, "col": col,
                           "flat": row*SHAPE[1]+col,
                           "first_report_time": str(min(points["time"][member])),
                           "raw_reports": int(len(member)),
                           "nearest_prior_report_m": float(np.min(nearest[member])),
                           "new_site": bool(np.min(nearest[member]) > radius)})
    return result


def build(candidate: np.ndarray, district_id: np.ndarray) -> dict[int, list[dict]]:
    if candidate.shape != SHAPE or district_id.shape != SHAPE:
        raise ValueError("City grid geometry changed")
    points = load_points()
    nearest = distance_to_prior_year(points)
    by_radius = {r: events(points, nearest, r, candidate, district_id)
                 for r in RADII}
    audit = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source": [str(p.relative_to(ROOT)) for p in SOURCES],
             "direct_category_points_inside_grid": len(points["day"]),
             "radii_m": RADII,
             "method": "Same-day complete-linkage report clusters; report medoid cell; "
                       "new if no point in cluster lies within radius of a report "
                       "from an earlier day in the preceding 365 days.",
             "q4": {str(r): {
                 "events": sum(e["day"] >= "2025-10-01" for e in es),
                 "new_events": sum(e["day"] >= "2025-10-01" and e["new_site"] for e in es),
             } for r, es in by_radius.items()},
             "limits": ["A report point is not a verified flood footprint",
                        "No report is not observed dry ground",
                        "Grouping and new-site status are radius-sensitive",
                        "Points on uncandidate boundary cells are omitted from events"]}
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2)+"\n")
    return by_radius


if __name__ == "__main__":
    from flood_citywide_history_ranker import load_grid
    data = load_grid()
    result = build(data["candidate"], data["district_id"])
    print(json.dumps({r: {"all": len(v), "q4_new": sum(e["day"] >= "2025-10-01" and
                           e["new_site"] for e in v)} for r, v in result.items()}, indent=2))
