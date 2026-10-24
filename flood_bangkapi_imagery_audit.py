"""Audit open satellite scene timing against 2025 BMA Bang Kapi road floods.

Run: .venv/bin/python flood_bangkapi_imagery_audit.py
STAC metadata checks scene availability, not local clear sky or flood extent.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "flood_bangkapi_imagery_audit"
OUTPUT = ROOT / "artifacts" / "flood_bangkapi_imagery_audit"
REPORT = ROOT / "data" / "flood_bangkok_transfer" / "bma_flood_2025.csv"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
BBOX = "100.595,13.735,100.675,13.825"  # All 13 named Bang Kapi points + margin.
BANGKOK = timezone(timedelta(hours=7))
COLLECTIONS = ("sentinel-1-rtc", "sentinel-2-l2a")


def report_intervals() -> dict[str, list[tuple[datetime, datetime]]]:
    by_date: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    with REPORT.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["district"].replace("เขต", "").strip() != "บางกะปิ":
                continue
            begin = datetime.strptime(
                row["frd_flood_date_begin"] + " " + row["frd_flood_begin"],
                "%m/%d/%Y %H:%M").replace(tzinfo=BANGKOK)
            end = datetime.strptime(
                row["frd_flood_date_end"] + " " + row["frd_flood_end"],
                "%m/%d/%Y %H:%M").replace(tzinfo=BANGKOK)
            if end < begin:
                raise ValueError(f"BMA interval ends before it begins: {row['frd_id']}")
            by_date[begin.date().isoformat()].append((begin, end))
    return dict(sorted(by_date.items()))


def scenes(collection: str) -> list[dict]:
    path = CACHE / f"{collection}_2025_bangkapi.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        response = requests.get(STAC, params={
            "collections": collection, "bbox": BBOX,
            "datetime": "2025-01-01/2025-12-31", "limit": 500}, timeout=120)
        response.raise_for_status()
        data = response.json()
        if any(link.get("rel") == "next" for link in data.get("links", [])):
            raise ValueError("STAC query was paginated; increase coverage handling")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False))
    return [{"id": item["id"],
             "time": datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")).astimezone(BANGKOK),
             "cloud_cover_scene_pct": item["properties"].get("eo:cloud_cover"),
             "orbit": item["properties"].get("sat:orbit_state")}
            for item in data["features"]]


def time_gap_hours(pass_time: datetime, intervals: list[tuple[datetime, datetime]]) -> float:
    return min(max((begin - pass_time).total_seconds(),
                   (pass_time - end).total_seconds(), 0) / 3600
               for begin, end in intervals)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    intervals = report_intervals()
    all_scenes = {collection: scenes(collection) for collection in COLLECTIONS}
    rows = []
    for day, reports in intervals.items():
        row = {"report_date": day, "report_rows": len(reports),
               "earliest_flood_begin": min(start for start, _ in reports).isoformat(timespec="minutes"),
               "latest_flood_end": max(end for _, end in reports).isoformat(timespec="minutes")}
        for collection, prefix in (("sentinel-1-rtc", "s1"),
                                   ("sentinel-2-l2a", "s2")):
            same_day = [scene for scene in all_scenes[collection]
                        if scene["time"].date().isoformat() == day]
            # A BMA flood can cross midnight, so check every scene, not only
            # scenes whose local date equals the report's start date.
            overlaps = [scene for scene in all_scenes[collection]
                        if time_gap_hours(scene["time"], reports) == 0]
            near = [scene for scene in all_scenes[collection]
                    if time_gap_hours(scene["time"], reports) <= 1]
            closest = min(all_scenes[collection],
                          key=lambda scene: time_gap_hours(scene["time"], reports))
            row.update({f"{prefix}_same_day_scenes": len(same_day),
                        f"{prefix}_pass_times_bkk": ";".join(scene["time"].isoformat(timespec="minutes") for scene in same_day),
                        f"{prefix}_overlap_scenes": len(overlaps),
                        f"{prefix}_within_1h_scenes": len(near),
                        f"{prefix}_nearest_gap_h": round(time_gap_hours(closest["time"], reports), 2),
                        f"{prefix}_nearest_scene_time_bkk": closest["time"].isoformat(timespec="minutes"),
                        f"{prefix}_same_day_cloud_cover_scene_pct": ";".join(
                            str(round(scene["cloud_cover_scene_pct"], 1))
                            for scene in same_day if scene["cloud_cover_scene_pct"] is not None)})
        rows.append(row)
    write_csv(OUTPUT / "scene_timing_by_report_date.csv", rows)
    summary = {"bbox": BBOX, "report_dates": len(rows),
               "collection_scene_counts_2025": {key: len(value) for key, value in all_scenes.items()},
               "s2_pre_september_scenes_below_10pct_scene_cloud": sum(
                   scene["time"].date().isoformat() < "2025-09-01"
                   and scene["cloud_cover_scene_pct"] is not None
                   and scene["cloud_cover_scene_pct"] < 10
                   for scene in all_scenes["sentinel-2-l2a"]),
               "dates_with_same_day_s1": sum(row["s1_same_day_scenes"] > 0 for row in rows),
               "dates_with_s1_during_reported_flood": sum(row["s1_overlap_scenes"] > 0 for row in rows),
               "dates_with_same_day_s2": sum(row["s2_same_day_scenes"] > 0 for row in rows),
               "dates_with_s2_during_reported_flood": sum(row["s2_overlap_scenes"] > 0 for row in rows),
               "dates_with_s2_scene_cloud_below_20pct": sum(
                   bool(row["s2_same_day_cloud_cover_scene_pct"]) and
                   any(float(v) < 20 for v in row["s2_same_day_cloud_cover_scene_pct"].split(";"))
                   for row in rows),
               "scope": "Scene metadata and reported intervals only; cloud percentage is for the satellite scene, not Bang Kapi."}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
