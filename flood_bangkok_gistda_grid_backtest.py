"""Align positive GISTDA water polygons to the 250 m Bangkok model grid.

The comparison is positive-only: top-queue capture of mapped water area.
Cells without a polygon retain `unknown` status, never `dry`.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shapely
from shapely.prepared import prep
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from flood_bangkok_spatial_cnn_citywide import FEATURE_CACHE
from flood_bangkok_transfer import load_districts
from flood_bangkapi_surface_grid import GRID_M, cell_box, grid_spec


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "flood_bangkapi_surface"
SCAN = OUTPUT / "gistda_2026_bma_daily_scan.json"
TO_UTM = Transformer.from_crs(4326, 32647, always_xy=True).transform
TO_LONLAT = Transformer.from_crs(32647, 4326, always_xy=True).transform
FRACTIONS = (0.05, 0.10, 0.20)
SCENARIO_CSV = OUTPUT / "surface_citywide_rain_scenario_stress_v1_cells.csv"
REFERENCE_MODELS = ("canal_cnn", "cnn_gnn")


def reference_scenario_scores() -> dict[str, dict[str, float]]:
    result = {model: {} for model in REFERENCE_MODELS}
    with SCENARIO_CSV.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["scenario"] == "20mm_2h" and row["model"] in result:
                cell_id = f"{row['district_code']}:{int(row['row']):02d}:{int(row['col']):02d}"
                result[row["model"]][cell_id] = float(row["proxy_score"])
    return result


def selection_weights(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Fractional allocation at a score tie; unbiased expected tie capture."""
    budget = fraction * len(scores)
    result = np.zeros(len(scores), dtype=float)
    used = 0.0
    for score in np.unique(scores)[::-1]:
        indices = np.flatnonzero(scores == score)
        share = min(1.0, max(0.0, (budget - used) / len(indices)))
        result[indices] = share
        used += share * len(indices)
        if used >= budget - 1e-9:
            break
    return result


def mapped_polygons(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    grouped = defaultdict(list)
    for feature in data["features"]:
        geometry = shapely.transform(shape(feature["geometry"]), TO_UTM,
                                     interleaved=False)
        if geometry.area > 0:
            grouped[feature["properties"]["district_code"]].append(geometry)
    return {code: unary_union(polygons) for code, polygons in grouped.items()}


def district_cells(district: dict) -> tuple[list[dict], np.ndarray]:
    code = district["code"]
    meta = json.loads((FEATURE_CACHE / f"{code}_meta.json").read_text())
    y = np.load(FEATURE_CACHE / f"{code}_y.npy", mmap_mode="r")
    rr = np.asarray(meta["valid_rows"])
    cc = np.asarray(meta["valid_cols"])
    if len(y) != 1461:
        raise ValueError(f"Unexpected history length: {code}")
    # This reference was available before the 2026 GISTDA products.
    historical = np.asarray(y[1096:1461, 0, rr, cc], dtype=np.uint8).sum(axis=0)
    spec = grid_spec(district["polygon"])
    cells = []
    for row, col in zip(rr, cc):
        geometry = cell_box(spec, int(row), int(col)).intersection(district["polygon"])
        cells.append({"day": "", "cell_id": f"{code}:{int(row):02d}:{int(col):02d}",
                      "district_code": code, "district": district["name_en"],
                      "row": int(row), "col": int(col),
                      "valid_cell_area_m2": geometry.area,
                      "historical_2025_complaint_days": 0,
                      "mapped_wet_area_m2": 0.0, "observation_status": "unknown",
                      "geometry": geometry})
    if len(cells) != len(historical):
        raise ValueError(f"Historical alignment failed: {code}")
    for cell, score in zip(cells, historical):
        cell["historical_2025_complaint_days"] = int(score)
    return cells, historical


def metrics(rows: list[dict], scores: dict[str, np.ndarray]) -> dict:
    wet = np.asarray([row["mapped_wet_area_m2"] for row in rows])
    total_area = float(wet.sum())
    touched = int(np.count_nonzero(wet > 1))
    result = {"candidate_cells": len(rows), "mapped_wet_cells": touched,
              "mapped_wet_area_km2": round(total_area / 1e6, 5)}
    for name, score in scores.items():
        item = {"zero_score_cells": int(np.count_nonzero(score == 0))}
        for fraction in FRACTIONS:
            selected = selection_weights(score, fraction)
            item[f"top_{int(fraction*100)}pct_expected_area_capture"] = (
                round(float(np.dot(selected, wet) / total_area), 4)
                if total_area > 0 else None)
            item[f"top_{int(fraction*100)}pct_expected_wet_cell_capture"] = (
                round(float(np.dot(selected, wet > 1) / touched), 4)
                if touched else None)
        result[name] = item
    return result


def main() -> None:
    scan = json.loads(SCAN.read_text())
    days = [entry["day"] for entry in scan["rows"]
            if entry["polygons_intersecting_bma"] > 0]
    districts = load_districts()
    reference = {d["code"]: district_cells(d) for d in districts}
    reference_scores = reference_scenario_scores()
    results = []
    for day in days:
        polygons = mapped_polygons(OUTPUT / f"gistda_daily_{day.replace('-', '')}_bma.geojson")
        rows = []
        district_results = []
        wet_features = []
        for district in districts:
            code = district["code"]
            base, score = reference[code]
            score_map = {"historical_2025": score.astype(float)}
            for model in REFERENCE_MODELS:
                scores = np.asarray([reference_scores[model][cell["cell_id"]]
                                     for cell in base], dtype=float)
                score_map[model + "_20mm_2h_reference"] = scores
            flood = polygons.get(code)
            prepared_flood = prep(flood) if flood is not None else None
            per_district = []
            for cell in base:
                item = {key: value for key, value in cell.items() if key != "geometry"}
                item["day"] = day
                for model in REFERENCE_MODELS:
                    item[model + "_20mm_2h_reference_score"] = (
                        reference_scores[model][cell["cell_id"]])
                if prepared_flood is not None and prepared_flood.intersects(cell["geometry"]):
                    area = flood.intersection(cell["geometry"]).area
                    if area > 1:
                        item["mapped_wet_area_m2"] = round(area, 2)
                        item["observation_status"] = "mapped_water"
                        wet_features.append({"type": "Feature", "geometry": mapping(
                            shapely.transform(cell["geometry"], TO_LONLAT,
                                              interleaved=False)),
                                             "properties": item.copy()})
                per_district.append(item)
            rows.extend(per_district)
            if flood is not None:
                summary = metrics(per_district, score_map)
                summary.update({"district_code": code, "district": district["name_en"],
                                "polygon_area_km2": round(flood.area / 1e6, 5),
                                "cell_area_vs_polygon_ratio": round(
                                    summary["mapped_wet_area_km2"] * 1e6 / flood.area, 4)})
                district_results.append(summary)
        csv_path = OUTPUT / f"gistda_{day.replace('-', '')}_grid_250m.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        geo_path = OUTPUT / f"gistda_{day.replace('-', '')}_wet_cells_250m.geojson"
        geo_path.write_text(json.dumps({"type": "FeatureCollection",
                                        "features": wet_features},
                                       separators=(",", ":")) + "\n")
        total_polygons = sum(polygon.area for polygon in polygons.values())
        total_cells = sum(row["mapped_wet_area_m2"] for row in rows)
        if abs(total_cells - total_polygons) / total_polygons > 0.02:
            raise ValueError(f"Mapped cell area differs from polygons by >2%: {day}")
        results.append({"day": day, "candidate_cells_citywide": len(rows),
                        "mapped_wet_cells_citywide": len(wet_features),
                        "polygon_area_km2": round(total_polygons / 1e6, 5),
                        "mapped_cell_area_km2": round(total_cells / 1e6, 5),
                        "districts": district_results,
                        "grid_csv": str(csv_path), "wet_cell_geojson": str(geo_path)})
        print(day, len(wet_features), round(total_cells / 1e6, 4), "km2", flush=True)
    summary = {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "grid_m": GRID_M, "label": "positive mapped water only",
               "unmapped_status": "unknown, not verified dry",
               "rankers": ["2025 per-cell direct Traffy complaint-positive days",
                           "saved canal CNN and CNN-GNN on fixed 1 Oct 2025 static scene with synthetic 20 mm/2 h rain"],
               "ranker_note": "Tie-aware fractional queue selection; saved model scores are spatial references, not 2026 date-matched forecasts; 2026 GISTDA extent has incomplete observation coverage",
               "selection": "Within each positive district, top 5/10/20% of all valid 250 m cells",
               "results": results}
    area_days = sum(entry["mapped_cell_area_km2"] for entry in results)
    aggregate = {"mapped_area_days_km2": round(area_days, 5),
                 "note": "Areas sum across dates, so repeated locations are counted more than once"}
    for name in ("historical_2025", "canal_cnn_20mm_2h_reference",
                 "cnn_gnn_20mm_2h_reference"):
        aggregate[name] = {}
        for fraction in FRACTIONS:
            key = f"top_{int(fraction*100)}pct_expected_area_capture"
            numerator = sum(d["mapped_wet_area_km2"] * d[name][key]
                            for entry in results for d in entry["districts"])
            aggregate[name][key] = round(numerator / area_days, 4)
    summary["positive_area_weighted_capture"] = aggregate
    output = OUTPUT / "gistda_2026_positive_only_grid_backtest.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("saved", output, flush=True)


if __name__ == "__main__":
    main()
