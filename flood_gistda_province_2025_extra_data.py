"""Add six spaced 2025 GISTDA product dates to the province training set.

These dates precede the fixed 2026 Bangkok evaluation. Empty products remain
unknown, not dry labels, and do not create training scenes.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

from flood_bangkok_gistda_extent_audit import public_example_key
from flood_bangkok_transfer import RAIN_URL
from flood_bangkapi_surface_model import RAIN_FEATURES, rainfall_by_date
from flood_gistda_multiregion_data import (
    CACHE, CENTERS, OUTPUT, SIZE, item_assets, one_label,
)
from flood_gistda_province_expand_data import patch_spec, selected_specs


SCAN = OUTPUT / "province_2025_candidate_scan.json"
AUDIT = OUTPUT / "province_2025_extra_data_audit.json"


def all_specs() -> dict:
    return {**{name: patch_spec(*center) for name, center in CENTERS.items()
               if name != "nong_chok"}, **selected_specs()}


def build_labels(specs: dict, dates: list[date], counts: dict) -> dict:
    pending = {}
    audits = {}
    for name in specs:
        path = CACHE / f"{name}_wet_2025_extra_250m.npz"
        if path.exists():
            with np.load(path) as saved:
                if list(saved["dates"].astype(str)) == [str(d) for d in dates]:
                    audits[name] = {"cached": True,
                                    "positive_dates": int(sum(x.sum() > 1 for x in saved["wet_area_m2"]))}
                    continue
        pending[name] = specs[name]
    if not pending:
        return audits
    assets = item_assets(dates, public_example_key())
    arrays = {name: np.zeros((len(dates), SIZE, SIZE), np.float32) for name in pending}
    details = {name: {} for name in pending}
    jobs = [(name, specs[name], day, assets[day]) for name in pending
            for day in dates if counts[name][str(day)] > 0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(one_label, job): (job[0], job[2]) for job in jobs}
        for future in as_completed(future_map):
            name, day, wet, audit = future.result()
            arrays[name][dates.index(day)] = wet
            details[name][str(day)] = audit
            print("2025 label", name, day, audit["mapped_wet_km2"], flush=True)
    for name, wet in arrays.items():
        np.savez_compressed(CACHE / f"{name}_wet_2025_extra_250m.npz",
                            wet_area_m2=wet, dates=np.asarray([str(d) for d in dates]))
        audits[name] = {"positive_dates": int(sum(x.sum() > 1 for x in wet)),
                        "mapped_area_days_km2": round(float(wet.sum()) / 1e6, 4),
                        "dates": details[name]}
    return audits


def build_rain(specs: dict, dates: list[date]) -> dict:
    audit = {}
    for name, spec in specs.items():
        path = CACHE / f"{name}_rain_2025_extra.json"
        lon, lat = spec["center_lonlat"]
        if not path.exists():
            response = requests.get(RAIN_URL, params={
                "latitude": lat, "longitude": lon,
                "start_date": str(min(dates) - timedelta(days=3)),
                "end_date": str(max(dates)),
                "hourly": "precipitation", "timezone": "Asia/Bangkok"}, timeout=70)
            if response.status_code != 200:
                raise RuntimeError(f"Open-Meteo {name} returned HTTP {response.status_code}")
            path.write_text(response.text)
        values = rainfall_by_date(min(dates), max(dates), path)
        matrix = np.asarray([[math.log1p(max(0, values[d][key]))
                              for key in RAIN_FEATURES] for d in dates], np.float32)
        np.savez_compressed(CACHE / f"{name}_rain_features_2025_extra.npz",
                            values=matrix, dates=np.asarray([str(d) for d in dates]))
        audit[name] = {"daily_rain_mm": {
            str(d): round(values[d]["rain_today_mm"], 2) for d in dates}}
        print("2025 rain", name, flush=True)
    return audit


def main() -> None:
    scan = json.loads(SCAN.read_text())
    dates = [date.fromisoformat(x) for x in scan["dates"]]
    specs = all_specs()
    if set(scan["bbox_matched_counts"]) != set(specs):
        raise ValueError("2025 scan does not cover training patches")
    coverage = build_labels(specs, dates, scan["bbox_matched_counts"])
    rain = build_rain(specs, dates)
    report = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "dates": [str(d) for d in dates], "regions": list(specs),
              "selection": "Six spaced 2025 catalog dates selected for this exploratory augmentation; Sep24 replaced by Sep25 because Sep24 has two STAC items. The 2026 Bangkok test was already inspected in earlier runs.",
              "labels": coverage, "rain": rain,
              "semantics": "positive GISTDA mapped area only; no polygon means unknown, not verified dry"}
    AUDIT.write_text(json.dumps(report, indent=2) + "\n")
    print("saved", AUDIT, flush=True)


if __name__ == "__main__":
    main()
