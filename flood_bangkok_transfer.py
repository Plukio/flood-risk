"""Bangkok-wide, district-transfer flood-report storm backtest and scenarios.

Run: .venv/bin/python flood_bangkok_transfer.py
Open data only. The target is an observed BMA report, not verified flooding.
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import requests
import shapefile
from pyproj import Transformer
from rasterio.features import geometry_mask
from scipy.stats import spearmanr
from shapely.geometry import shape
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "flood_bangkok_transfer"
OUTPUT = ROOT / "artifacts" / "flood_bangkok_transfer"
DISTRICT_URL = "https://data.bangkok.go.th/dataset/1e04f888-6287-41ce-aaa8-91f3bc6dae25/resource/1e870bdb-6c1c-4900-ac66-e293248cdeb1/download/district.zip"
DTM_URL = "https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer/exportImage"
RAIN_URL = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_META_URL = "https://data.bangkok.go.th/api/3/action/package_show?id=frd_dds"
TO_LONLAT = Transformer.from_crs(32647, 4326, always_xy=True)
RAIN_THRESHOLD_MM_H = 0.5
MAX_DRY_GAP_H = 3
HOLDOUT_MODULUS = 5  # District codes divisible by five are never used for fitting.


def fetch(path: Path, url: str, params: dict | None = None) -> bytes:
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=120,
                                    headers={"User-Agent": "Bangkok open-data flood research"})
            response.raise_for_status()
            if len(response.content) < 100:
                raise ValueError(f"Unexpectedly short response: {response.text[:100]}")
            path.write_bytes(response.content)
            return response.content
        except (requests.RequestException, ValueError) as error:
            last_error = error
            time.sleep(attempt + 1)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_districts() -> list[dict]:
    archive = zipfile.ZipFile(io.BytesIO(fetch(CACHE / "bma_district.zip", DISTRICT_URL)))
    reader = shapefile.Reader(
        shp=io.BytesIO(archive.read("district.shp")),
        shx=io.BytesIO(archive.read("district.shx")),
        dbf=io.BytesIO(archive.read("district.dbf")), encoding="cp874")
    districts = []
    for record in reader.shapeRecords():
        properties = record.record.as_dict()
        polygon = shape(record.shape.__geo_interface__)
        point = polygon.representative_point()
        lon, lat = TO_LONLAT.transform(point.x, point.y)
        districts.append({"code": properties["dcode"],
                          "name_th": properties["dname"].replace("เขต", "").strip(),
                          "name_en": properties["dname_e"].strip(),
                          "lon": lon, "lat": lat,
                          "area_km2": polygon.area / 1_000_000,
                          "polygon": polygon,
                          "spatial_holdout": int(properties["dcode"]) % HOLDOUT_MODULUS == 0})
    assert len(districts) == 50 and len({d["code"] for d in districts}) == 50
    return sorted(districts, key=lambda d: d["code"])


def terrain_for_district(district: dict) -> dict:
    name = district["code"]
    polygon = district["polygon"]
    left, bottom, right, top = polygon.bounds
    params = {"bbox": f"{left},{bottom},{right},{top}", "bboxSR": 32647,
              "imageSR": 32647, "size": "512,512", "format": "tiff",
              "pixelType": "F32", "f": "image"}
    raw = fetch(CACHE / f"terrain_{name}.tif", DTM_URL, params)
    with rasterio.MemoryFile(raw) as memory:
        with memory.open() as raster:
            data = raster.read(1).astype(float)
            inside = geometry_mask([polygon.__geo_interface__], data.shape,
                                   raster.transform, invert=True)
    valid = inside & np.isfinite(data) & (data > -20) & (data < 60)
    if np.sum(valid) < 1000 or np.sum(valid) / max(1, np.sum(inside)) < 0.9:
        raise ValueError(f"Terrain coverage too low for {name}")
    q10, q50, q90 = np.percentile(data[valid], [10, 50, 90])
    return {"elev_p10_m": round(float(q10), 3),
            "elev_median_m": round(float(q50), 3),
            "elev_relief_m": round(float(q90 - q10), 3),
            "terrain_valid_fraction": round(float(np.sum(valid) / np.sum(inside)), 4)}


def weather_for_district(district: dict) -> dict:
    params = {"latitude": district["lat"], "longitude": district["lon"],
              "start_date": "2021-01-01", "end_date": "2025-12-31",
              "hourly": "precipitation", "timezone": "Asia/Bangkok"}
    data = json.loads(fetch(CACHE / f"rain_{district['code']}_2021_2025.json", RAIN_URL, params))
    rainfall = data["hourly"]["precipitation"]
    valid = sum(value is not None for value in rainfall) / len(rainfall)
    if valid < 0.95:
        raise ValueError(f"Rain coverage only {valid:.1%} for {district['name_en']}")
    return data


def make_storms(district: dict, weather: dict) -> list[dict]:
    timestamps = [datetime.fromisoformat(t) for t in weather["hourly"]["time"]]
    rain = np.array([np.nan if v is None else float(v) for v in weather["hourly"]["precipitation"]])
    active = np.flatnonzero(rain >= RAIN_THRESHOLD_MM_H)
    if len(active) == 0:
        return []
    boundaries = []
    start = last = int(active[0])
    for idx in active[1:]:
        idx = int(idx)
        if idx - last - 1 > MAX_DRY_GAP_H:
            boundaries.append((start, last))
            start = idx
        last = idx
    boundaries.append((start, last))
    storms = []
    for first, last in boundaries:
        segment = rain[first:last + 1]
        if np.any(~np.isfinite(segment)):
            continue
        previous = rain[max(0, first - 24):first]
        if np.any(~np.isfinite(previous)):
            continue
        start_time = timestamps[first]
        storms.append({"district_code": district["code"],
                       "district_th": district["name_th"],
                       "district_en": district["name_en"],
                       "spatial_holdout": int(district["spatial_holdout"]),
                       "storm_start": start_time.isoformat(timespec="minutes"),
                       "storm_end": (timestamps[last] + timedelta(hours=1)).isoformat(timespec="minutes"),
                       "year": start_time.year, "month": start_time.month,
                       "rain_total_mm": round(float(np.sum(segment)), 3),
                       "rain_peak_1h_mm": round(float(np.max(segment)), 3),
                       "rain_duration_h": last - first + 1,
                       "rain_prev24h_mm": round(float(np.sum(previous)), 3),
                       "elev_p10_m": district["elev_p10_m"],
                       "elev_median_m": district["elev_median_m"],
                       "elev_relief_m": district["elev_relief_m"],
                       "area_km2": round(district["area_km2"], 3),
                       "reported": 0, "report_rows": 0})
    return storms


def load_reports() -> list[dict]:
    reports = []
    metadata = json.loads(fetch(CACHE / "bma_flood_metadata.json", FLOOD_META_URL))["result"]
    for year in range(2021, 2026):
        resource = next(r for r in metadata["resources"] if f"({year})" in r.get("name", ""))
        data = fetch(CACHE / f"bma_flood_{year}.csv", resource["url"])
        for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
            try:
                when = datetime.strptime(
                    row["frd_flood_date_begin"] + " " + row["frd_flood_begin"],
                    "%m/%d/%Y %H:%M")
            except ValueError:
                continue
            reports.append({"district_th": row["district"].replace("เขต", "").strip(),
                            "when": when, "road": row["frd_road"], "area": row["frd_area"]})
    return reports


def attach_reports(storms: list[dict], reports: list[dict]) -> dict:
    by_district = defaultdict(list)
    for storm in storms:
        by_district[storm["district_th"]].append(storm)
    matched = 0
    unmatched_by_district = Counter()
    for report in reports:
        candidates = []
        for storm in by_district[report["district_th"]]:
            start = datetime.fromisoformat(storm["storm_start"])
            end = datetime.fromisoformat(storm["storm_end"])
            if not start - timedelta(hours=3) <= report["when"] <= end + timedelta(hours=6):
                continue
            distance = max((start - report["when"]).total_seconds(),
                           (report["when"] - end).total_seconds(), 0)
            candidates.append((distance, abs((start - report["when"]).total_seconds()), storm))
        if candidates:
            storm = min(candidates, key=lambda x: (x[0], x[1]))[2]
            storm["report_rows"] += 1
            storm["reported"] = 1
            matched += 1
        else:
            unmatched_by_district[report["district_th"]] += 1
    return {"all_report_rows": len(reports), "matched_report_rows": matched,
            "match_fraction": round(matched / max(1, len(reports)), 4),
            "unmatched_by_district": dict(unmatched_by_district)}


def vector(row: dict, terrain: bool) -> list[float]:
    values = [math.log1p(float(row[key])) for key in
              ("rain_total_mm", "rain_peak_1h_mm", "rain_duration_h", "rain_prev24h_mm")]
    if terrain:
        values += [float(row["elev_p10_m"]), float(row["elev_median_m"]),
                   float(row["elev_relief_m"]), math.log1p(float(row["area_km2"]))]
    month = int(row["month"])
    values += [math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)]
    return values


def fit_model(rows: list[dict], terrain: bool) -> HistGradientBoostingClassifier:
    data = [row for row in rows if row["year"] <= 2024 and not row["spatial_holdout"]]
    x = np.array([vector(row, terrain) for row in data])
    y = np.array([row["reported"] for row in data])
    monotone = [1, 1, 1, 1] + ([0] * 4 if terrain else []) + [0, 0]
    model = HistGradientBoostingClassifier(
        loss="log_loss", max_iter=150, max_leaf_nodes=7,
        min_samples_leaf=100, learning_rate=0.05, l2_regularization=5,
        monotonic_cst=monotone, random_state=7)
    model.fit(x, y)
    return model


def score_rows(model: HistGradientBoostingClassifier, rows: list[dict], terrain: bool) -> np.ndarray:
    return model.predict_proba(np.array([vector(row, terrain) for row in rows]))[:, 1]


def event_metrics(rows: list[dict], scores: np.ndarray) -> dict:
    y = np.array([row["reported"] for row in rows])
    return {"storms": len(rows), "reported_storms": int(np.sum(y)),
            "report_rate": round(float(np.mean(y)), 5),
            "average_precision": round(float(average_precision_score(y, scores)), 5),
            "roc_auc": round(float(roc_auc_score(y, scores)), 5) if len(np.unique(y)) == 2 else None,
            "brier": round(float(brier_score_loss(y, scores)), 5)}


def ranking_agreement(predicted: list[float], observed: list[float]) -> dict:
    if len(predicted) < 3 or len(set(observed)) < 2:
        return {"spearman": None, "pairwise_agreement": None, "comparable_pairs": 0}
    comparable = 0
    agreement = 0.0
    for i in range(len(observed)):
        for j in range(i + 1, len(observed)):
            truth = np.sign(observed[i] - observed[j])
            if truth == 0:
                continue
            comparable += 1
            guess = np.sign(predicted[i] - predicted[j])
            agreement += 0.5 if guess == 0 else float(guess == truth)
    return {"spearman": round(float(spearmanr(predicted, observed).statistic), 4),
            "pairwise_agreement": round(agreement / comparable, 4),
            "comparable_pairs": comparable}


def make_scenario(district: dict, rate: float, hours: int, antecedent: float,
                  month: int) -> dict:
    return {"rain_total_mm": rate * hours, "rain_peak_1h_mm": rate,
            "rain_duration_h": hours, "rain_prev24h_mm": antecedent,
            "elev_p10_m": district["elev_p10_m"],
            "elev_median_m": district["elev_median_m"],
            "elev_relief_m": district["elev_relief_m"],
            "area_km2": district["area_km2"], "month": month}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-mm-h", type=float, default=4)
    parser.add_argument("--duration-h", type=int, default=2)
    parser.add_argument("--antecedent-mm", type=float, default=0)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--district", type=str, default="",
                        help="Print one district's scenario row after processing all 50")
    args = parser.parse_args()
    if args.rate_mm_h <= 0 or args.duration_h <= 0 or args.antecedent_mm < 0:
        parser.error("Rain rate and duration must be positive; antecedent rain cannot be negative")
    if not 1 <= args.month <= 12:
        parser.error("Month must be between 1 and 12")
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    districts = load_districts()
    with ThreadPoolExecutor(max_workers=5) as executor:
        terrain_jobs = {executor.submit(terrain_for_district, d): d for d in districts}
        for job in as_completed(terrain_jobs):
            terrain_jobs[job].update(job.result())
    print("Loaded terrain for 50 districts", flush=True)
    weather_data = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        weather_jobs = {executor.submit(weather_for_district, d): d for d in districts}
        for job in as_completed(weather_jobs):
            district = weather_jobs[job]
            weather = job.result()
            weather_data[district["code"]] = weather
            district["rain_grid_lat"] = weather["latitude"]
            district["rain_grid_lon"] = weather["longitude"]
    print("Loaded hourly rain for 50 districts", flush=True)
    storms = [storm for district in districts
              for storm in make_storms(district, weather_data[district["code"]])]
    reports = load_reports()
    report_audit = attach_reports(storms, reports)
    rain_model = fit_model(storms, terrain=False)
    terrain_model = fit_model(storms, terrain=True)
    train = [r for r in storms if r["year"] <= 2024 and not r["spatial_holdout"]]
    seen_2025 = [r for r in storms if r["year"] == 2025 and not r["spatial_holdout"]]
    transfer_2025 = [r for r in storms if r["year"] == 2025 and r["spatial_holdout"]]
    assert len({r["district_code"] for r in transfer_2025}) == 10
    for group in (seen_2025, transfer_2025):
        p_rain = score_rows(rain_model, group, False)
        p_terrain = score_rows(terrain_model, group, True)
        for row, a, b in zip(group, p_rain, p_terrain):
            row["p_report_rain_only"] = round(float(a), 6)
            row["p_report_rain_terrain"] = round(float(b), 6)
    evaluations = {
        "seen_districts_2025": {"rain_only": event_metrics(seen_2025, np.array([r["p_report_rain_only"] for r in seen_2025])),
                                "rain_terrain": event_metrics(seen_2025, np.array([r["p_report_rain_terrain"] for r in seen_2025]))},
        "unseen_districts_2025": {"rain_only": event_metrics(transfer_2025, np.array([r["p_report_rain_only"] for r in transfer_2025])),
                                  "rain_terrain": event_metrics(transfer_2025, np.array([r["p_report_rain_terrain"] for r in transfer_2025]))},
    }
    district_ranks = []
    raw_2025 = Counter(r["district_th"] for r in reports if r["when"].year == 2025)
    for district in districts:
        rows = [r for r in storms if r["district_code"] == district["code"] and r["year"] == 2025]
        district_ranks.append({"district_code": district["code"], "district": district["name_en"],
                               "held_out_from_training": int(district["spatial_holdout"]),
                               "storm_count_2025": len(rows),
                               "predicted_report_storms_rain": round(sum(r["p_report_rain_only"] for r in rows), 3),
                               "predicted_report_storms_terrain": round(sum(r["p_report_rain_terrain"] for r in rows), 3),
                               "observed_matched_report_storms": sum(r["reported"] for r in rows),
                               "observed_bma_incident_rows": raw_2025[district["name_th"]]})
    district_ranks.sort(key=lambda row: -row["predicted_report_storms_terrain"])
    for i, row in enumerate(district_ranks, 1):
        row["predicted_rank_terrain"] = i
    for group_name, held_out in (("all_50", None), ("unseen_10", 1)):
        subset = [r for r in district_ranks if held_out is None or r["held_out_from_training"] == held_out]
        for prediction in ("predicted_report_storms_rain", "predicted_report_storms_terrain"):
            key = "rain_only" if prediction == "predicted_report_storms_rain" else "rain_terrain"
            evaluations.setdefault("district_ranking_2025", {}).setdefault(group_name, {})[key] = ranking_agreement(
                [r[prediction] for r in subset], [r["observed_matched_report_storms"] for r in subset])
    training_rain = np.array([[r["rain_total_mm"], r["rain_peak_1h_mm"], r["rain_duration_h"]] for r in train])
    p99 = np.percentile(training_rain, 99, axis=0)
    scenarios = []
    for district in districts:
        variants = {"base": (1, 1), "double_intensity": (2, 1),
                    "double_duration": (1, 2), "double_both": (2, 2)}
        row = {"district_code": district["code"], "district": district["name_en"],
               "held_out_from_training": int(district["spatial_holdout"])}
        for name, (intensity, duration) in variants.items():
            sample = make_scenario(district, args.rate_mm_h * intensity,
                                   args.duration_h * duration, args.antecedent_mm,
                                   args.month)
            row[f"p_report_{name}"] = round(float(score_rows(terrain_model, [sample], True)[0]), 6)
            row[f"outside_training_p99_{name}"] = int(
                sample["rain_total_mm"] > p99[0] or sample["rain_peak_1h_mm"] > p99[1]
                or sample["rain_duration_h"] > p99[2])
        row["delta_double_both"] = round(row["p_report_double_both"] - row["p_report_base"], 6)
        assert row["p_report_double_intensity"] + 1e-6 >= row["p_report_base"]
        assert row["p_report_double_duration"] + 1e-6 >= row["p_report_base"]
        assert row["p_report_double_both"] + 1e-6 >= max(
            row["p_report_double_intensity"], row["p_report_double_duration"])
        scenarios.append(row)
    scenarios.sort(key=lambda row: -row["delta_double_both"])
    write_csv(OUTPUT / "district_catalog.csv", [
        {k: v for k, v in d.items() if k != "polygon"} for d in districts])
    write_csv(OUTPUT / "storms.csv", storms)
    write_csv(OUTPUT / "district_ranking_2025.csv", district_ranks)
    write_csv(OUTPUT / "scenario_districts.csv", scenarios)
    result = {"target": "BMA report during an Open-Meteo-defined rain event; no report is unobserved",
              "spatial_unit": "Bangkok district, not street",
              "scenario_interpretation": "model response under imposed constant hourly rain, not verified physical flood probability",
              "train_storms": len(train), "train_reported_storms": sum(r["reported"] for r in train),
              "districts": len(districts), "unseen_districts": [d["name_en"] for d in districts if d["spatial_holdout"]],
              "distinct_rain_grids": len({(d["rain_grid_lat"], d["rain_grid_lon"]) for d in districts}),
              "report_matching": report_audit, "evaluations": evaluations,
              "scenario_input": {"base_rate_mm_h": args.rate_mm_h, "base_duration_h": args.duration_h,
                                 "antecedent_mm": args.antecedent_mm, "month": args.month},
              "training_rain_p99": {"event_total_mm": round(float(p99[0]), 2),
                                    "peak_1h_mm": round(float(p99[1]), 2),
                                    "duration_h": round(float(p99[2]), 2)}}
    (OUTPUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("report_matching", "evaluations")}, indent=2))
    print("Transfer test:", json.dumps(evaluations["unseen_districts_2025"], indent=2))
    if args.district:
        selected = [row for row in scenarios if args.district.casefold() in row["district"].casefold()]
        if not selected:
            parser.error(f"No Bangkok district matched {args.district!r}")
        print("Selected district scenarios:", json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
