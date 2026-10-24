"""Open-data Lak Si flood-report hindcast. Run with: .venv/bin/python flood_laksi_backtest.py

The target is a BMA *report* at a previously known site on a calendar day.
An absent report is not proof that the road was dry. This is a data-readiness
backtest, not a validated flood-probability model.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "flood_laksi"
OUTPUT = ROOT / "artifacts" / "flood_laksi"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BKK flood research prototype (open-data backtest)"})
THAI_TZ = timezone(timedelta(hours=7))
YEARS = range(2021, 2026)
PILOT_DISTRICT = "หลักสี่"
FLOOD_DATASET = "https://data.bangkok.go.th/api/3/action/package_show?id=frd_dds"
DTM_SERVICE = "https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer/exportImage"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
S1_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

# All six site names occurred in BMA reports before the held-out year 2025.
# Coordinates are used only to extract terrain, not to select test locations.
# Location reference: BMA flood_risk26 point inventory (published 2026).
SITES = {
    "lotus": {"label": "Chaeng Watthana, Lotus", "lat": 13.895008, "lon": 100.555700},
    "m11": {"label": "Chaeng Watthana, M.T.B. 11", "lat": 13.894381, "lon": 100.556657},
    "pongphet": {"label": "Ngam Wong Wan, Pongphet", "lat": 13.856001, "lon": 100.545931},
    "chinkhet": {"label": "Ngam Wong Wan, Chin Khet", "lat": 13.857836, "lon": 100.548682},
    "court": {"label": "Chaeng Watthana, Administrative Court", "lat": 13.890214, "lon": 100.568372},
    "soi10": {"label": "Chaeng Watthana Soi 10", "lat": 13.887480, "lon": 100.578930},
}


def cached_request(path: Path, url: str, params: dict | None = None) -> bytes:
    if path.exists():
        return path.read_bytes()
    response = SESSION.get(url, params=params, timeout=90)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_flood_rows() -> list[dict]:
    metadata = json.loads(cached_request(CACHE / "bma_flood_metadata.json", FLOOD_DATASET))["result"]
    result = []
    for resource in metadata["resources"]:
        name = resource.get("name", "")
        matched = next((year for year in YEARS if f"({year})" in name), None)
        if matched is None:
            continue
        raw = cached_request(CACHE / f"bma_flood_{matched}.csv", resource["url"])
        for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
            if row["district"].replace("เขต", "").strip() == PILOT_DISTRICT:
                row["year"] = matched
                result.append(row)
    return result


def site_for_report(row: dict) -> str | None:
    area = re.sub(r"\s+", "", row["frd_area"])
    if "โลตัส" in area:
        return "lotus"
    if any(word in area for word in ("มทบ", "มณฑลทหาร", "กรมทหาร")):
        return "m11"
    if "พงษ์เพชร" in area:
        return "pongphet"
    if "ชินเขต" in area:
        return "chinkhet"
    if any(word in area for word in ("ศาลปกครอง", "ศาลปกคลอง")):
        return "court"
    if any(word in area for word in ("ซอย10", "แจ้งวัฒนะ10")):
        return "soi10"
    return None


def parse_bma_datetime(row: dict, date_field: str, time_field: str) -> datetime:
    return datetime.strptime(
        f"{row[date_field]} {row[time_field]}", "%m/%d/%Y %H:%M"
    ).replace(tzinfo=THAI_TZ)


def get_terrain() -> dict[str, dict]:
    transformer = Transformer.from_crs(4326, 32647, always_xy=True)
    terrain = {}
    for site, meta in SITES.items():
        x, y = transformer.transform(meta["lon"], meta["lat"])
        params = {
            "bbox": f"{x-128},{y-128},{x+128},{y+128}",
            "bboxSR": 32647,
            "imageSR": 32647,
            "size": "256,256",
            "format": "tiff",
            "pixelType": "F32",
            "f": "image",
        }
        raw = cached_request(CACHE / f"dtm_{site}_256m.tif", DTM_SERVICE, params)
        with rasterio.MemoryFile(raw) as memory:
            with memory.open() as dataset:
                values = dataset.read(1).astype(float)
        finite = np.isfinite(values) & (values > -20) & (values < 60)
        if np.mean(finite) < 0.95:
            raise ValueError(f"Too much missing or implausible terrain at {site}")
        cy, cx = np.indices(values.shape)
        radius = np.hypot(cx - 127.5, cy - 127.5)
        centre = float(np.median(values[finite & (radius <= 10)]))
        ring = values[finite & (radius >= 40) & (radius <= 110)]
        relative = centre - float(np.median(ring))
        terrain[site] = {
            "elevation_m": round(centre, 3),
            "relative_to_ring_m": round(relative, 3),
            "depression_m": round(max(0.0, np.percentile(ring, 25) - centre), 3),
            "terrain_valid_fraction": round(float(np.mean(finite)), 4),
        }
    return terrain


def get_rain(site: str, lat: float, lon: float) -> dict[date, dict]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2021-01-01",
        "end_date": "2025-12-31",
        "hourly": "precipitation",
        "timezone": "Asia/Bangkok",
    }
    data = json.loads(cached_request(CACHE / f"openmeteo_best_match_{site}_2021_2025.json", OPEN_METEO, params))
    if "hourly" not in data:
        raise ValueError(f"No historical rain data for {site}: {data}")
    raw_values = data["hourly"]["precipitation"]
    valid_fraction = sum(v is not None for v in raw_values) / max(1, len(raw_values))
    if valid_fraction < 0.95:
        raise ValueError(f"Historical rain is only {valid_fraction:.1%} complete at {site}")
    hours = [(datetime.fromisoformat(t), None if v is None else float(v))
             for t, v in zip(data["hourly"]["time"], raw_values)]
    by_date: dict[date, list[float]] = defaultdict(list)
    for timestamp, mm in hours:
        by_date[timestamp.date()].append(mm)
    dates = sorted(by_date)
    result = {}
    for day in dates:
        rain = by_date[day]
        if len(rain) != 24 or any(value is None for value in rain):
            continue
        prev1_values = by_date.get(day - timedelta(days=1), [])
        prev3_values = [value for i in (1, 2, 3)
                        for value in by_date.get(day - timedelta(days=i), [])]
        if any(value is None for value in prev1_values + prev3_values):
            continue
        prev1 = sum(prev1_values)
        prev3 = sum(prev3_values)
        result[day] = {
            "rain_24h": round(sum(rain), 3),
            "rain_peak_1h": round(max(rain), 3),
            "rain_peak_3h": round(max(sum(rain[i:i+3]) for i in range(22)), 3),
            "rain_prev_24h": round(prev1, 3),
            "rain_prev_72h": round(prev3, 3),
        }
    return result


def sar_overlap(rows: list[dict]) -> dict:
    params = {
        "collections": "sentinel-1-grd",
        "bbox": "100.54,13.85,100.60,13.90",
        "datetime": "2021-01-01/2025-12-31",
        "limit": 500,
    }
    data = json.loads(cached_request(CACHE / "sentinel1_laksi_2021_2025.json", S1_STAC, params))
    scene_times = [datetime.fromisoformat(feature["properties"]["datetime"].replace("Z", "+00:00"))
                   .astimezone(THAI_TZ) for feature in data["features"]]
    intervals = []
    for row in rows:
        try:
            begin = parse_bma_datetime(row, "frd_flood_date_begin", "frd_flood_begin")
            end = parse_bma_datetime(row, "frd_flood_date_end", "frd_flood_end")
            if end >= begin:
                intervals.append((begin, end))
        except ValueError:
            continue
    answer = {"scenes": len(scene_times), "reports_with_valid_intervals": len(intervals),
              "distinct_report_dates": len({begin.date() for begin, _ in intervals})}
    for window_hours in (0, 1, 3):
        margin = timedelta(hours=window_hours)
        dates = {begin.date() for begin, end in intervals
                 if any(begin - margin <= scene <= end + margin for scene in scene_times)}
        answer[f"matched_dates_plusminus_{window_hours}h"] = len(dates)
    return answer


def make_features(reports: list[dict], terrain: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    labeled = set()
    matches = []
    for report in reports:
        site = site_for_report(report)
        try:
            day = parse_bma_datetime(report, "frd_flood_date_begin", "frd_flood_begin").date()
        except ValueError:
            day = None
        matches.append({"year": report["year"], "date": day.isoformat() if day else "",
                        "road": report["frd_road"], "area": report["frd_area"],
                        "matched_site": site or "", "reported_rain_mm": report["frd_total_rainfall"]})
        if site and day:
            labeled.add((site, day))
    feature_rows = []
    for site, meta in SITES.items():
        rain = get_rain(site, meta["lat"], meta["lon"])
        for day, wx in sorted(rain.items()):
            row = {"site": site, "date": day.isoformat(), "year": day.year,
                   "reported_flood": int((site, day) in labeled), **wx, **terrain[site]}
            feature_rows.append(row)
    return feature_rows, matches


RAIN_COLS = ("rain_24h", "rain_peak_1h", "rain_peak_3h", "rain_prev_24h", "rain_prev_72h")
TERRAIN_COLS = ("elevation_m", "relative_to_ring_m", "depression_m")


def vector(row: dict, mode: str) -> list[float]:
    values = [math.log1p(max(0.0, float(row[col]))) for col in RAIN_COLS]
    month = int(row["date"][5:7])
    values += [math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)]
    if mode in {"rain_site", "rain_terrain"}:
        if mode == "rain_site":
            values += [float(row["site"] == site) for site in SITES]
        else:
            values += [float(row[col]) for col in TERRAIN_COLS]
    return values


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "samples": int(len(y)), "reports": int(np.sum(y)),
        "report_rate": round(float(np.mean(y)), 5),
        "average_precision": round(float(average_precision_score(y, score)), 5),
        "roc_auc": round(float(roc_auc_score(y, score)), 5) if len(np.unique(y)) == 2 else None,
        "brier": round(float(brier_score_loss(y, score)), 5),
    }


def backtest(rows: list[dict]) -> tuple[dict, list[dict]]:
    train = [row for row in rows if row["year"] <= 2024]
    test = [row for row in rows if row["year"] == 2025]
    assert train and test and max(row["year"] for row in train) < min(row["year"] for row in test)
    y_train = np.array([row["reported_flood"] for row in train])
    y_test = np.array([row["reported_flood"] for row in test])
    scores = {}
    model_metrics = {}
    for mode in ("rain", "rain_site", "rain_terrain"):
        x_train = np.array([vector(row, mode) for row in train])
        x_test = np.array([vector(row, mode) for row in test])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=1000))
        model.fit(x_train, y_train)
        scores[mode] = model.predict_proba(x_test)[:, 1]
        model_metrics[mode] = metrics(y_test, scores[mode])
    site_counts = Counter(row["site"] for row in train if row["reported_flood"])
    site_days = Counter(row["site"] for row in train)
    prior = np.array([(site_counts[row["site"]] + 1) / (site_days[row["site"]] + 2)
                      for row in test])
    model_metrics["historical_site_rate"] = metrics(y_test, prior)
    rain_rank = np.array([row["rain_peak_3h"] for row in test], dtype=float)
    # Ranking-only baseline: no calibrated probability or Brier score.
    rain_metric = metrics(y_test, rain_rank / max(1.0, float(np.max(rain_rank))))
    rain_metric["brier"] = None
    model_metrics["raw_peak_3h_ranking"] = rain_metric
    wet_mask = np.array([row["rain_24h"] >= 1.0 for row in test])
    wet_metrics = {name: metrics(y_test[wet_mask], score[wet_mask])
                   for name, score in scores.items()}
    predictions = [{"date": row["date"], "site": row["site"],
                    "reported_flood": row["reported_flood"],
                    "rain_24h": row["rain_24h"], "rain_peak_3h": row["rain_peak_3h"],
                    "p_report_rain": round(float(scores["rain"][i]), 6),
                    "p_report_rain_site": round(float(scores["rain_site"][i]), 6),
                    "p_report_rain_terrain": round(float(scores["rain_terrain"][i]), 6)}
                   for i, row in enumerate(test)]
    return {"training_site_days": len(train), "training_reports": int(np.sum(y_train)),
            "test_site_days": len(test), "test_reports": int(np.sum(y_test)),
            "models": model_metrics, "wet_day_models": wet_metrics,
            "test_reports_with_grid_rain_below_1mm": int(np.sum(y_test[~wet_mask]))}, predictions


def street_ranking(predictions: list[dict], reports: list[dict]) -> tuple[list[dict], dict]:
    """Rank the six pre-2025 road locations, then compare with held-out outcomes."""
    pre2025 = Counter(site_for_report(row) for row in reports if row["year"] < 2025)
    durations = defaultdict(list)
    depths = defaultdict(list)
    for report in reports:
        if report["year"] != 2025:
            continue
        site = site_for_report(report)
        if not site:
            continue
        try:
            hours, minutes = map(int, report["frd_flood_total"].split(":"))
            durations[site].append(hours + minutes / 60)
        except ValueError:
            pass
        try:
            depths[site].append(float(report["frd_flood_height"]))
        except ValueError:
            pass
    by_site = defaultdict(list)
    for prediction in predictions:
        by_site[prediction["site"]].append(prediction)
    rows = []
    for site, daily in by_site.items():
        rows.append({
            "site": site,
            "street_location": SITES[site]["label"],
            "historical_reports_2021_2024": pre2025[site],
            "expected_report_days_rain": round(sum(p["p_report_rain"] for p in daily), 4),
            "expected_report_days_rain_site": round(sum(p["p_report_rain_site"] for p in daily), 4),
            "expected_report_days_rain_terrain": round(sum(p["p_report_rain_terrain"] for p in daily), 4),
            "observed_report_days_2025": sum(p["reported_flood"] for p in daily),
            "observed_flood_hours_2025": round(sum(durations[site]), 2),
            "observed_mean_depth_cm_2025": round(float(np.mean(depths[site])), 2) if depths[site] else "",
        })
    rows.sort(key=lambda row: (-row["expected_report_days_rain_site"], row["site"]))
    observed_days = np.array([row["observed_report_days_2025"] for row in rows], dtype=float)
    observed_hours = np.array([row["observed_flood_hours_2025"] for row in rows], dtype=float)
    actual_ranks = rankdata(-observed_days, method="min")
    for i, (row, rank) in enumerate(zip(rows, actual_ranks), start=1):
        row["predicted_rank_rain_site"] = i
        row["observed_rank_by_report_days"] = int(rank)
    rank_metrics = {}
    for method, column in {
        "rain_only": "expected_report_days_rain",
        "rain_site": "expected_report_days_rain_site",
        "rain_terrain": "expected_report_days_rain_terrain",
        "historical_reports": "historical_reports_2021_2024",
    }.items():
        score = np.array([row[column] for row in rows], dtype=float)
        comparable = 0
        agreement = 0.0
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                truth = np.sign(observed_days[i] - observed_days[j])
                if truth == 0:
                    continue
                comparable += 1
                predicted = np.sign(score[i] - score[j])
                agreement += 0.5 if predicted == 0 else float(predicted == truth)
        rank_metrics[method] = {
            "spearman_report_days": round(float(spearmanr(score, observed_days).statistic), 4),
            "spearman_flood_hours": round(float(spearmanr(score, observed_hours).statistic), 4),
            "pairwise_agreement_report_days": round(agreement / comparable, 4),
            "comparable_pairs": comparable,
        }
    return rows, rank_metrics


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = get_flood_rows()
    terrain = get_terrain()
    features, matches = make_features(reports, terrain)
    result, predictions = backtest(features)
    ranking, rank_metrics = street_ranking(predictions, reports)
    result["street_ranking_2025"] = rank_metrics
    result["target_definition"] = "BMA road-flood report at a recurring Lak Si site on a calendar day; no report is unobserved, not confirmed dry"
    result["scenario_definition"] = "Retrospective full-day rain hindcast, not an advance forecast"
    result["rain_source"] = "Open-Meteo Historical Weather API best_match hourly precipitation"
    result["incident_rows_laksi"] = len(reports)
    result["incident_rows_matched"] = sum(bool(row["matched_site"]) for row in matches)
    result["incident_rows_matched_2025"] = sum(row["year"] == 2025 and bool(row["matched_site"]) for row in matches)
    rain_grids = {}
    for site in SITES:
        weather = json.loads((CACHE / f"openmeteo_best_match_{site}_2021_2025.json").read_text())
        rain_grids[site] = {"lat": weather["latitude"], "lon": weather["longitude"]}
    result["rain_grids"] = rain_grids
    result["distinct_rain_grids"] = len({(v["lat"], v["lon"]) for v in rain_grids.values()})
    daily = {(row["site"], row["date"]): row for row in features}
    compared = []
    for match in matches:
        if match["year"] != 2025 or not match["matched_site"]:
            continue
        observed = daily.get((match["matched_site"], match["date"]))
        if observed is None:
            continue
        try:
            compared.append((float(match["reported_rain_mm"]), float(observed["rain_24h"])))
        except ValueError:
            continue
    if compared:
        result["rain_comparison_2025"] = {
            "matched_report_rows": len(compared),
            "median_bma_reported_event_mm": round(float(np.median([v[0] for v in compared])), 2),
            "median_openmeteo_grid_calendar_day_mm": round(float(np.median([v[1] for v in compared])), 2),
            "warning": "Different places and time windows; this is a resolution diagnostic, not a bias correction.",
        }
    result["sites"] = {site: {**meta, **terrain[site]} for site, meta in SITES.items()}
    result["sentinel1_overlap"] = sar_overlap(reports)
    (OUTPUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUTPUT / "incident_matches.csv", matches, list(matches[0]))
    write_csv(OUTPUT / "daily_features.csv", features, list(features[0]))
    write_csv(OUTPUT / "predictions_2025.csv", predictions, list(predictions[0]))
    write_csv(OUTPUT / "street_ranking_2025.csv", ranking, list(ranking[0]))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
