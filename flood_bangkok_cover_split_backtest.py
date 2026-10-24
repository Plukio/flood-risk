"""Compare separate WorldCover classes on the Bangkok flood-report target.

Outcome: Traffy flood-report buffer touching a 250 m cell-day, not independently
verified wet/dry water. Uses the matched Bang Kapi/Bang Khae training protocol.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from flood_bangkok_new_feature_ablation import GROUPS, load_group
from flood_bangkok_river_level_backtest import river_features, river_levels
from flood_bangkok_spatial_cnn_multiyear import TRAIN_END
from flood_bangkok_spatial_water_state import (
    CODES, load_district, matrix, model, training_indices,
)
from flood_bangkok_transfer import CACHE
from flood_bangkapi_surface_grid import OUTPUT
from flood_bangkapi_surface_model import rainfall_by_date, rank_metrics
from flood_gistda_multiregion_worldcover import CHANNELS


VARIANTS = ("old_cover", "split_cover")
SPLIT_INDICES = (1, 2, 3, 4)  # crop, tree/grass, bare, wetland


def features(info: dict, di: np.ndarray, ci: np.ndarray, variant: str) -> np.ndarray:
    old = info["new_static"][ci][:, GROUPS["surface_cover"]]
    parts = [matrix(info, di, ci, "baseline"), old]
    if variant == "split_cover":
        parts.append(info["worldcover_split"][ci][:, SPLIT_INDICES])
    return np.column_stack(parts).astype(np.float32)


def fold(data: dict, days: list[date], train_end: date,
         test_start: date, test_end: date) -> dict:
    n_train = sum(day <= train_end for day in days)
    first, last = days.index(test_start), days.index(test_end) + 1
    output = {}
    for label, source, target in (("bang_khae_local", "1040", "1040"),
                                  ("bang_kapi_to_bang_khae", "1006", "1040")):
        src = data[source]
        dst = {**data[target], "prior": data[target]["prior"].copy()}
        dst["prior"][first:last] = dst["prior"][first]
        train_di, train_ci, weights = training_indices(src["y"], n_train)
        test_di = np.repeat(np.arange(first, last), dst["cells"])
        test_ci = np.tile(np.arange(dst["cells"]), last - first)
        truth = dst["y"][first:last].astype(np.uint8)
        output[label] = {}
        for variant in VARIANTS:
            x_train = features(src, train_di, train_ci, variant)
            x_test = features(dst, test_di, test_ci, variant)
            fitted = model().fit(x_train, src["y"][train_di, train_ci],
                                 sample_weight=weights)
            prob = fitted.predict_proba(x_test)[:, 1].reshape(truth.shape)
            output[label][variant] = {
                "cell_rank": rank_metrics(prob.sum(axis=0), truth.sum(axis=0)),
                "daily_average_precision": round(float(average_precision_score(
                    truth.ravel(), prob.ravel())), 5),
                "daily_brier": round(float(brier_score_loss(
                    truth.ravel(), prob.ravel())), 6),
                "predicted_positive_cell_days": round(float(prob.sum()), 1),
                "observed_positive_cell_days": int(truth.sum()),
                "train_positives": int(src["y"][train_di, train_ci].sum()),
            }
            print(label, variant, output[label][variant], flush=True)
    return output


def main() -> None:
    levels, _ = river_levels()
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   CACHE / "rain_1006_2021_2025.json"))
    river = river_features(days, levels)
    data = {}
    for code in CODES:
        info = load_district(code, days, river)
        with np.load(OUTPUT / f"surface_shared_state_{code}_daily_cells.npz") as saved:
            rr, cc = saved["rows"], saved["cols"]
        info["new_static"], old_names = load_group(code, "new_static", rr, cc)
        info["worldcover_split"], cover_names = load_group(
            code, "worldcover_split", rr, cc)
        if tuple(cover_names) != CHANNELS:
            raise ValueError(f"WorldCover channel order changed: {code}")
        if ("worldcover_built_fraction_2021" not in old_names or
                "worldcover_water_fraction_2021" not in old_names):
            raise ValueError("Old WorldCover fields missing")
        for index in (0, 5):
            old = info["new_static"][:, old_names.index(CHANNELS[index])]
            new = info["worldcover_split"][:, index]
            if np.max(np.abs(old - new)) > 1e-5:
                raise ValueError(f"WorldCover extraction disagrees: {code}, {index}")
        data[code] = info
    result = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "Traffy flood-report-positive 250 m cell-day; missing report is not verified dry",
        "comparison": "Original surface-cover group versus same group plus separate 2021 crop, tree/grass, bare, and wetland fractions",
        "source": "ESA WorldCover 2021; existing OSM and weather features",
        "splits": {
            "2025": fold(data, days, TRAIN_END, date(2025, 10, 1), date(2025, 12, 31)),
            "2024": fold(data, days, date(2024, 7, 31), date(2024, 10, 1), date(2024, 12, 31)),
        },
        "limits": ["2021 land cover applied to 2022–25; 2025 OSM applied to earlier years",
                   "Both test seasons have been inspected in earlier experiments",
                   "A report absence is not a field-verified dry observation"],
    }
    (OUTPUT / "surface_cover_split_backtest_metrics.json").write_text(json.dumps(
        result, indent=2) + "\n")


if __name__ == "__main__":
    main()
