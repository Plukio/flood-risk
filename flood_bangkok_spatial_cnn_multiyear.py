"""Compare multi-year Bang Kapi and six-district spatial CNN training.

All models use the same 250 m cells, 100 m direct Traffy flood-report buffer,
17 terrain/history/season/rain features plus mask, and October–December 2025
Bang Kapi test. This target is a report proxy, not observed inundation.

Run flood_bangkok_traffy_history.py first, then:
    .venv/bin/python flood_bangkok_spatial_cnn_multiyear.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkok_spatial_cnn_crossdistrict import (
    ARTIFACT as SINGLE_YEAR_ARTIFACT, CHANNELS, DISTRICTS,
    fit, normalize, report_labels, static_grid,
)
from flood_bangkok_traffy_history import MORE_CODES
from flood_bangkapi_spatial_cnn import percentiles
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import (
    DIRECT_TYPES, OUTPUT, RAIN_FEATURES, TRAFFY, rainfall_by_date, rank_metrics, write_csv,
)


ROOT = Path(__file__).resolve().parent
OLD_BANG_KAPI = ROOT / "data/flood_bangkapi_surface/traffy_flood_2022_2024_minimal.csv"
OLD_OTHER = TRANSFER_CACHE / "traffy_flood_2022_2024_five_districts.csv"
OLD_EXPANDED = TRANSFER_CACHE / "traffy_flood_2022_2024_ten_districts.csv"
ARTIFACT = "surface_crossdistrict_terrain_cnn_multiyear"
TRAIN_END = date(2025, 7, 31)
CAL_END = date(2025, 9, 30)


def report_rows(expanded: bool = False) -> dict[str, dict[int, list[dict]]]:
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in (OLD_BANG_KAPI, OLD_EXPANDED if expanded else OLD_OTHER, TRAFFY):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run the Traffy history extractors")
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row["type"] in DIRECT_TYPES:
                    grouped[row["district"]][int(row["timestamp"][:4])].append(row)
    return grouped


def district_data(district: dict, reports: dict, stations: list[dict],
                  height: int, width: int):
    spec, static, valid, cells = static_grid(district, stations)
    labels = defaultdict(set)
    audits = {}
    for year in range(2022, 2026):
        annual, audits[str(year)] = report_labels(
            district, spec, cells, reports.get(district["name_th"], {}).get(year, []), year)
        for address, days in annual.items():
            labels[address].update(days)
    rain = rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                            TRANSFER_CACHE / f"rain_{district['code']}_2021_2025.json")
    days = sorted(rain)
    if len(days) != 1461:
        raise ValueError("Expected complete 2022–2025 daily rain")
    data = np.zeros((len(days), CHANNELS + 1, height, width), dtype=np.float32)
    target = np.zeros((len(days), 1, height, width), dtype=np.float32)
    h, w = valid.shape
    data[:, CHANNELS, :h, :w] = valid
    data[:, :7, :h, :w] = static
    addresses = list(cells)
    rr = np.asarray([r for r, _ in addresses])
    cc = np.asarray([c for _, c in addresses])
    prior = Counter()
    for t, day in enumerate(days):
        if t and day.year != days[t - 1].year:
            prior.clear()
        seasonal = [math.sin(2 * math.pi * day.month / 12),
                    math.cos(2 * math.pi * day.month / 12)]
        weather = [math.log1p(max(0, rain[day][key])) for key in RAIN_FEATURES]
        for row, col in addresses:
            data[t, 7:CHANNELS, row, col] = [prior[(row, col)], *seasonal, *weather]
            target[t, 0, row, col] = int(day in labels.get((row, col), set()))
        if day.year < 2025 or day <= CAL_END:
            for address in addresses:
                prior[address] += int(day in labels.get(address, set()))
    return data, target, (rr, cc), audits


def main(expanded: bool = False) -> None:
    import torch

    torch.set_num_threads(4)
    codes = DISTRICTS + MORE_CODES if expanded else DISTRICTS
    artifact = f"{ARTIFACT}_eleven" if expanded else ARTIFACT
    districts = {d["code"]: d for d in load_districts() if d["code"] in codes}
    if len(districts) != len(codes):
        raise ValueError("Missing selected BMA district")
    specs = {code: grid_spec(districts[code]["polygon"]) for code in codes}
    height = max(spec["rows"] for spec in specs.values())
    width = max(spec["cols"] for spec in specs.values())
    reports = report_rows(expanded)
    stations = pump_points()
    district_arrays = {}
    audits = {}
    for code in codes:
        data, target, addresses, audit = district_data(
            districts[code], reports, stations, height, width)
        district_arrays[code] = data, target, addresses
        audits[code] = {"name": districts[code]["name_en"], "years": audit}
        print(f"district {code} {districts[code]['name_en']}: "
              f"{[(year, item[f'positive_cell_days_{year}']) for year, item in audit.items()]}",
              flush=True)
    bk_x, bk_y, (rr, cc) = district_arrays["1006"]
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   TRANSFER_CACHE / "rain_1006_2021_2025.json"))
    train_count = sum(day <= TRAIN_END for day in days)
    cal_end = sum(day <= CAL_END for day in days)
    if train_count != 1308 or cal_end != 1369:
        raise ValueError("Unexpected temporal split")
    if bk_y[cal_end:, 0, rr, cc].sum() != 242:
        raise ValueError("Bang Kapi test labels differ from the 100 m baseline")
    results = {}
    predictions = {}
    observed = None
    variants = (("eleven_districts_multiyear", codes),) if expanded else (
        ("bang_kapi_multiyear", ("1006",)), ("six_districts_multiyear", DISTRICTS))
    for name, training_codes in variants:
        train_x = np.concatenate([district_arrays[code][0][:train_count] for code in training_codes])
        train_y = np.concatenate([district_arrays[code][1][:train_count] for code in training_codes])
        x_cal, y_cal = bk_x[train_count:cal_end], bk_y[train_count:cal_end]
        x_test, y_test = bk_x[cal_end:], bk_y[cal_end:]
        train_norm, cal_norm, test_norm = normalize(train_x, x_cal, x_test)
        result, expected, observed = fit(train_norm, train_y, cal_norm, y_cal,
                                         test_norm, y_test, rr, cc)
        result["training_district_codes"] = list(training_codes)
        result["training_positive_cell_days"] = int(train_y.sum())
        results[name] = result
        predictions[name] = expected
        print(name, result["test_cell_rank"],
              result["test_daily"]["average_precision"], flush=True)
        del train_x, train_y, train_norm, cal_norm, test_norm
    historical = bk_y[days.index(date(2025, 1, 1)):cal_end, 0, rr, cc].sum(axis=0)
    results["historical_baseline"] = rank_metrics(historical, observed)
    baseline = json.loads((OUTPUT / f"{SINGLE_YEAR_ARTIFACT}_metrics.json").read_text())
    if any(baseline["models"][name]["test_observed_positive_cell_days"] != 242
           for name in ("bang_kapi_only", "six_districts")):
        raise ValueError("Single-year baseline uses different test labels")
    with (OUTPUT / f"{SINGLE_YEAR_ARTIFACT}_cell_comparison.csv").open(
            encoding="utf-8-sig", newline="") as file:
        old_rows = {row["cell_id"]: row for row in csv.DictReader(file)}
    previous_rows = None
    if expanded:
        with (OUTPUT / f"{ARTIFACT}_cell_comparison.csv").open(
                encoding="utf-8-sig", newline="") as file:
            previous_rows = {row["cell_id"]: row for row in csv.DictReader(file)}
    rows = []
    for i in range(len(rr)):
        cid = f"bk:{int(rr[i]):02d}:{int(cc[i]):02d}"
        old = old_rows[cid]
        if int(old["observed_days_oct_dec"]) != observed[i]:
            raise ValueError("Single-year comparison has different cell outcome")
        row = {"cell_id": cid,
                     "historical_positive_days_jan_sep": int(historical[i]),
                     "bk_2025_only_expected_days": old["bk_only_cnn_expected_days_oct_dec"],
                     "six_districts_2025_only_expected_days": old[
                         "six_district_cnn_expected_days_oct_dec"],
                     "observed_days_oct_dec": int(observed[i]),
                     "historical_rank_percentile": round(float(percentiles(historical)[i]), 2)}
        if expanded:
            previous = previous_rows[cid]
            if int(previous["observed_days_oct_dec"]) != observed[i]:
                raise ValueError("Six-district multiyear comparison has different cell outcome")
            row.update({"bk_multiyear_expected_days": previous["bk_multiyear_expected_days"],
                        "six_districts_multiyear_expected_days": previous[
                            "six_districts_multiyear_expected_days"],
                        "eleven_districts_multiyear_expected_days": round(float(predictions[
                            "eleven_districts_multiyear"][i]), 5),
                        "eleven_districts_multiyear_rank_percentile": round(float(percentiles(
                            predictions["eleven_districts_multiyear"])[i]), 2)})
        else:
            row.update({"bk_multiyear_expected_days": round(float(predictions[
                            "bang_kapi_multiyear"][i]), 5),
                        "six_districts_multiyear_expected_days": round(float(predictions[
                            "six_districts_multiyear"][i]), 5),
                        "bk_multiyear_rank_percentile": round(float(percentiles(predictions[
                            "bang_kapi_multiyear"])[i]), 2),
                        "six_districts_multiyear_rank_percentile": round(float(percentiles(predictions[
                            "six_districts_multiyear"])[i]), 2)})
        rows.append(row)
    write_csv(OUTPUT / f"{artifact}_cell_comparison.csv", rows)
    metrics = {"target": "100 m Traffy direct flood-report buffer touches a 250 m cell-day",
               "not_target": "verified surface inundation",
               "train_dates": ["2022-01-01", "2025-07-31"],
               "selection_and_calibration_dates": ["2025-08-01", "2025-09-30"],
               "test_dates": ["2025-10-01", "2025-12-31"], "test_area": "Bang Kapi",
               "districts": audits, "models": results,
               "single_year_model_reference": f"{SINGLE_YEAR_ARTIFACT}_metrics.json",
               "six_district_multiyear_reference": f"{ARTIFACT}_metrics.json" if expanded else None,
               "input_features": baseline["input_features"],
               "limitations": ["October–December 2025 inspected in earlier experiments; exploratory test",
                               "single district-point gridded rain per district",
                               "present-day pump/gate locations applied to older years",
                               "report absence is not verified dry"]}
    (OUTPUT / f"{artifact}_metrics.json").write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: result.get("test_cell_rank", result)
                      for name, result in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", action="store_true",
                        help="Train on Bang Kapi and ten other districts")
    main(parser.parse_args().expanded)
