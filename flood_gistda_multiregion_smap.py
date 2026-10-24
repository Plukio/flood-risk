"""Rasterize publicly cataloged weekly GISTDA SMAP products for 2026 dates.

Only an item created before local midnight on the GISTDA product day is used.
The October 20 input is the last listed October 5 product and is flagged stale
in the audit; these values are coarse antecedent wetness, not 250 m readings.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import requests
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import shape

from flood_bangkok_gistda_extent_audit import public_example_key
from flood_gistda_multiregion_data import CACHE, GRID_M, OUTPUT, SIZE, TO_UTM, specs


COLLECTIONS = "https://api-gateway.gistda.or.th/api/2.0/resources/stac/drought/collections"
WEEKS = (date(2026, 9, 21), date(2026, 9, 28), date(2026, 10, 5))
CHANNEL = "gistda_smap_last_available_mean"


def item_for(day: date, key: str) -> dict:
    url = f"{COLLECTIONS}/smap_{day:%Y%m%d}/items"
    response = requests.get(url, params={"api_key": key, "limit": 10}, timeout=35)
    if response.status_code != 200:
        raise RuntimeError(f"SMAP collection {day} returned HTTP {response.status_code}")
    features = response.json().get("features", [])
    if len(features) != 1 or "data" not in features[0].get("assets", {}):
        raise ValueError(f"Expected one SMAP data asset on {day}")
    return features[0]


def features_for(item: dict, spec: dict) -> tuple[list[dict], int]:
    response = requests.get(item["assets"]["data"]["href"], params={
        "bbox": ",".join(map(str, spec["query_bbox"])), "limit": 10000,
        "offset": 0}, timeout=90)
    if response.status_code != 200:
        raise RuntimeError(f"SMAP data request returned HTTP {response.status_code}")
    result = response.json()
    found = result.get("features", [])
    matched = result.get("numberMatched", -1)
    if matched != len(found):
        raise ValueError(f"SMAP query paginated or incomplete: {matched}, {len(found)}")
    return found, matched


def surface(item: dict, spec: dict) -> tuple[np.ndarray, dict]:
    features, matched = features_for(item, spec)
    west, _, _, north = spec["utm_bounds"]
    shapes = []
    for feature in features:
        value = feature.get("properties", {}).get("smap_mean")
        if value is None or not np.isfinite(value):
            continue
        geometry = shapely.transform(shape(feature["geometry"]),
                                     TO_UTM.transform, interleaved=False)
        shapes.append((geometry, float(value)))
    array = rasterize(shapes, out_shape=(SIZE, SIZE),
                      transform=from_origin(west, north, GRID_M, GRID_M),
                      fill=np.nan, dtype="float32")
    coverage = float(np.isfinite(array).mean())
    if coverage < .90:
        raise ValueError(f"SMAP coverage too low: {coverage:.3f}")
    # Edge slivers can miss the cell center. Use a local median for rare gaps;
    # their fraction is preserved in the audit.
    median = float(np.nanmedian(array))
    array = np.where(np.isfinite(array), array, median).astype(np.float32)
    return array, {"matched_polygons": matched,
                   "rasterized_polygons": len(shapes),
                   "valid_cell_fraction": round(coverage, 5),
                   "mean": round(float(array.mean()), 5),
                   "min": round(float(array.min()), 5),
                   "max": round(float(array.max()), 5)}


def main() -> None:
    key = public_example_key()
    weekly_items = {day: item_for(day, key) for day in WEEKS}
    event_days = None
    audit = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source": "https://data.go.th/dataset/disaster-platform-drought",
             "collection_dates": [str(day) for day in WEEKS],
             "channel": CHANNEL, "interpretation": "coarse antecedent moisture, not observed 250 m soil moisture",
             "patches": {}}
    for name, spec in specs().items():
        with np.load(CACHE / f"{name}_wet_250m.npz") as saved:
            dates = [date.fromisoformat(str(x)) for x in saved["dates"]]
        if event_days is None:
            event_days = dates
        elif event_days != dates:
            raise ValueError("GISTDA dates differ across patches")
        weekly = {}
        coverage = {}
        for day, item in weekly_items.items():
            weekly[day], coverage[str(day)] = surface(item, spec)
        layers = []
        choices = []
        for event in dates:
            event_start = datetime.combine(event, time.min, ZoneInfo("Asia/Bangkok"))
            eligible = [day for day, item in weekly_items.items()
                        if datetime.fromisoformat(item["properties"]["created"].replace("Z", "+00:00"))
                        < event_start and day <= event]
            if not eligible:
                raise ValueError(f"No as-of SMAP item for {event}")
            day = max(eligible)
            layers.append(weekly[day])
            choices.append({"event_date": str(event), "source_date": str(day),
                            "age_days": (event - day).days,
                            "source_created_utc": weekly_items[day]["properties"]["created"]})
        values = np.stack(layers)[:, None].astype(np.float32)
        np.savez_compressed(CACHE / f"{name}_smap_2026_250m.npz",
                            values=values, names=np.asarray((CHANNEL,)),
                            dates=np.asarray([str(day) for day in dates]))
        audit["patches"][name] = {"weekly": coverage, "as_of_event": choices}
        print(name, coverage, flush=True)
    (OUTPUT / "smap_2026_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
