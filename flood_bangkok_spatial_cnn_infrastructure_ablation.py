"""Separate canal geometry from mapped assets in the 42-district CNN.

Run after flood_bangkok_spatial_cnn_infrastructure.py:
    .venv/bin/python flood_bangkok_spatial_cnn_infrastructure_ablation.py
"""

from __future__ import annotations

import json

import torch

from flood_bangkok_spatial_cnn_citywide import (
    ARTIFACT as BASE_ARTIFACT, CHANNELS, HOLDOUT, evaluate, prepare_cache, train_model,
)
from flood_bangkok_spatial_cnn_infrastructure import (
    ARTIFACT as INFRA_ARTIFACT, augmented_arrays,
)
from flood_bangkok_transfer import load_districts
from flood_bangkapi_surface_model import OUTPUT, write_csv


ARTIFACT = "surface_crossdistrict_infrastructure_ablation_citywide_v1"
MASKED = {
    "canal_only": tuple(range(CHANNELS + 2, CHANNELS + 7)),
    "assets_only": (CHANNELS, CHANNELS + 1),
}


def main() -> None:
    torch.set_num_threads(4)
    districts = {d["code"]: d for d in load_districts()}
    meta = prepare_cache(districts)
    arrays = augmented_arrays(meta)
    training_codes = tuple(code for code in districts if code not in HOLDOUT)
    base = json.loads((OUTPUT / f"{BASE_ARTIFACT}_metrics.json").read_text())
    all_features = json.loads((OUTPUT / f"{INFRA_ARTIFACT}_metrics.json").read_text())
    baseline_rows = {r["code"]: r for r in base["models"][
        "forty_two_more_steps"]["evaluated_districts"]}
    all_rows = {r["code"]: r for r in all_features["evaluated_districts"]}
    results = {}
    by_variant = {}
    for name, masked in MASKED.items():
        model, stats, train = train_model(
            name, training_codes, arrays, meta["1006"], max_epochs=40,
            patience=18, artifact=ARTIFACT, masked_channels=masked)
        rows = evaluate(model, stats, arrays, meta, tuple(districts),
                        masked_channels=masked)
        results[name] = {"training": train, "evaluated_districts": rows}
        by_variant[name] = {r["code"]: r for r in rows}
    eleven = set(base["models"]["eleven_matched"]["training"]["training_codes"])
    comparison = []
    for code, district in districts.items():
        old, full = baseline_rows[code], all_rows[code]
        if old["test_positive_cell_days"] != full["test_positive_cell_days"]:
            raise ValueError(f"Outcome mismatch in {code}")
        row = {"code": code, "district": district["name_en"],
               "group": "geographic_holdout" if code in HOLDOUT else
                        "original_eleven" if code in eleven else "added_training",
               "test_positive_cell_days": old["test_positive_cell_days"],
               "historical_spearman": old["historical_spearman"],
               "baseline_spearman": old["cnn_spearman"],
               "all_infrastructure_spearman": full["cnn_spearman"],
               "baseline_daily_ap": old["test_daily_ap"],
               "all_infrastructure_daily_ap": full["test_daily_ap"]}
        for name in MASKED:
            item = by_variant[name][code]
            if item["test_positive_cell_days"] != old["test_positive_cell_days"]:
                raise ValueError(f"Ablation outcome mismatch in {code}")
            row[name + "_spearman"] = item["cnn_spearman"]
            row[name + "_daily_ap"] = item["test_daily_ap"]
            row[name + "_predicted_positive_cell_days"] = item[
                "predicted_positive_cell_days"]
        comparison.append(row)
    write_csv(OUTPUT / f"{ARTIFACT}_district_comparison.csv", comparison)
    payload = {"target": all_features["target"],
               "train_period": all_features["train_period"],
               "selection_and_local_calibration": all_features[
                   "selection_and_local_calibration"],
               "test_period": all_features["test_period"],
               "geographic_holdout_codes": list(HOLDOUT),
               "feature_groups": {
                   "canal_only": "OSM canal length and distance; other five new channels masked",
                   "assets_only": "pump capacity, pond storage/distance, underpass presence/distance; canal channels masked"},
               "baseline_metrics": str(OUTPUT / f"{BASE_ARTIFACT}_metrics.json"),
               "all_infrastructure_metrics": str(OUTPUT / f"{INFRA_ARTIFACT}_metrics.json"),
               "models": results,
               "limitations": all_features["limitations"]}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("saved", ARTIFACT, flush=True)


if __name__ == "__main__":
    main()
