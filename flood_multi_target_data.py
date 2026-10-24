"""Align open Bangkok report and matched BMA targets to satellite patch grids.

Missing reports and missing BMA depths remain unknown, never dry or zero depth.
The cached complaint target is a 100 m complaint buffer touching a 250 m cell.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta, datetime, timezone

import numpy as np
from shapely.ops import unary_union

from flood_bangkok_multitask_depth_cnn import CODES, matched_depths
from flood_bangkok_spatial_cnn_citywide import FEATURE_CACHE
from flood_bangkok_transfer import CACHE as BMA_CACHE, load_districts
from flood_bangkapi_surface_grid import GRID_M, TO_UTM, grid_spec
from flood_bangkapi_surface_model import RAIN_FEATURES, rainfall_by_date
from flood_open_weak_bangkok_holdout_eval import bma_fraction
from flood_open_weak_data import OUTPUT, ROOT, all_specs


START = date(2022, 1, 1)
END = date(2025, 9, 30)  # Keep Q4 2025 reports and depths out of training.
PATCH_CODES = {"bang_kapi_core": "1006", "nong_chok": "1003"}
DEST = ROOT / "data" / "flood_multi_target"
AUDIT = OUTPUT / "multi_target_data_audit.json"
SHAPE = (80, 80)


def dates() -> list[date]:
    return [START + timedelta(days=i) for i in range((END - START).days + 1)]


def weather_for(code: str, days: list[date]) -> np.ndarray:
    path = BMA_CACHE / f"rain_{code}_2021_2025.json"
    observed = rainfall_by_date(START, END, path)
    values = []
    for day in days:
        row = [math.log1p(max(0., observed[day][key])) for key in RAIN_FEATURES]
        values.append(row + [math.sin(2*math.pi*day.month/12),
                             math.cos(2*math.pi*day.month/12)])
    return np.asarray(values, np.float32)


def complaint_grid(name: str, spec: dict, districts: list[dict], n_days: int):
    west, south, east, north = spec["utm_bounds"]
    result = np.zeros((n_days, *SHAPE), np.uint8)
    audit = []
    for district in districts:
        grid = grid_spec(district["polygon"])
        if (grid["east"] <= west or grid["west"] >= east or
                grid["north"] <= south or grid["south"] >= north):
            continue
        path = FEATURE_CACHE / f"{district['code']}_y.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        source = np.load(path, mmap_mode="r")
        if source.shape[0] != 1461:
            raise ValueError(f"Unexpected report dates in {path}")
        hits = np.argwhere(source[:n_days, 0] > 0)
        if not len(hits):
            continue
        rr = np.rint((north - (grid["north"] - hits[:, 1]*GRID_M)) / GRID_M).astype(int)
        cc = np.rint((grid["west"] + hits[:, 2]*GRID_M - west) / GRID_M).astype(int)
        valid = (rr >= 0) & (rr < SHAPE[0]) & (cc >= 0) & (cc < SHAPE[1])
        result[hits[valid, 0], rr[valid], cc[valid]] = 1
        audit.append({"district_code": district["code"], "name": district["name_en"],
                      "source_positive_cell_days": int(len(hits)),
                      "inside_patch_before_dedup": int(valid.sum())})
    return result, audit


def build() -> dict:
    DEST.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    all_days = dates()
    day_index = {day: i for i, day in enumerate(all_days)}
    patches = all_specs()
    districts = load_districts()
    _, rows, match_audit = matched_depths({d["code"]: d for d in districts
                                           if d["code"] in CODES})
    bma_polygon = unary_union([d["polygon"] for d in districts])
    output = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "period": [str(START), str(END)],
              "report_target": "100 m Traffy complaint buffer touches a 250 m cell-day",
              "depth_target": "BMA reported cm at a uniquely name-matched 2026 inventory point; approximate historical cell",
              "rain_target": "retrospective full local calendar-day; may include rain after report",
              "matched_bma_source_audit": match_audit,
              "patches": {}}
    for name, code in PATCH_CODES.items():
        spec = patches[name]
        reports, contributors = complaint_grid(name, spec, districts, len(all_days))
        mask = bma_fraction(spec, bma_polygon) > .02
        reports[:, ~mask] = 0
        depths = np.full((len(all_days), *SHAPE), np.nan, np.float32)
        west, south, east, north = spec["utm_bounds"]
        accepted = []
        for row in rows:
            day = date.fromisoformat(row["date"])
            if day not in day_index:
                continue
            x, y = TO_UTM.transform(row["inventory_lon"], row["inventory_lat"])
            if not (west <= x < east and south < y <= north):
                continue
            rr = math.floor((north - y) / GRID_M)
            cc = math.floor((x - west) / GRID_M)
            if not (0 <= rr < 80 and 0 <= cc < 80 and mask[rr, cc]):
                continue
            i = day_index[day]
            current = depths[i, rr, cc]
            depths[i, rr, cc] = max(float(row["depth_cm"]), current) if np.isfinite(current) else row["depth_cm"]
            accepted.append({"date": str(day), "row": rr, "col": cc,
                             "depth_cm": row["depth_cm"], "report_id": row["report_id"]})
        weather = weather_for(code, all_days)
        if not np.all(np.isfinite(weather)) or np.isfinite(depths).sum() > len(accepted):
            raise ValueError(f"Invalid target arrays for {name}")
        path = DEST / f"{name}_2022_2025sep.npz"
        np.savez_compressed(path, reports=reports, depth_cm=depths,
                            bma_mask=mask.astype(np.uint8), weather=weather,
                            dates=np.asarray([str(day) for day in all_days]))
        output["patches"][name] = {
            "path": str(path.relative_to(ROOT)), "rain_district_code": code,
            "days": len(all_days), "candidate_cells": int(mask.sum()),
            "report_positive_cell_days": int(reports.sum()),
            "report_positive_days": int((reports.sum(axis=(1, 2)) > 0).sum()),
            "matched_bma_rows_in_patch": len(accepted),
            "unique_depth_cell_days": int(np.isfinite(depths).sum()),
            "matched_depth_records": accepted,
            "contributing_districts": contributors}
        print(name, output["patches"][name]["report_positive_cell_days"],
              "report cell-days;", output["patches"][name]["unique_depth_cell_days"],
              "depth cell-days", flush=True)
    AUDIT.write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    build()
