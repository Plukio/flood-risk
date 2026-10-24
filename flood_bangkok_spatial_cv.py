"""Five-fold geographic transfer audit for flood_bangkok_transfer.py.

Run after the main script: .venv/bin/python flood_bangkok_spatial_cv.py
Every 2025 district is scored by a model trained on the other 40 districts'
2021–2024 storms. No 2025 data or same-district labels enter its fit.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from flood_bangkok_transfer import event_metrics, make_scenario, ranking_agreement, vector


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts" / "flood_bangkok_transfer"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(rows: list[dict], terrain: bool) -> HistGradientBoostingClassifier:
    x = np.array([vector(row, terrain) for row in rows])
    y = np.array([row["reported"] for row in rows])
    constraints = [1] * 4 + ([0] * 4 if terrain else []) + [0, 0]
    return HistGradientBoostingClassifier(
        loss="log_loss", max_iter=150, max_leaf_nodes=7,
        min_samples_leaf=100, learning_rate=0.05,
        l2_regularization=5, monotonic_cst=constraints,
        random_state=7).fit(x, y)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--district", default="",
                        help="Print one district's out-of-district scenario after scoring all 50")
    args = parser.parse_args()
    with (ARTIFACTS / "storms.csv").open(encoding="utf-8-sig") as file:
        storms = list(csv.DictReader(file))
    with (ARTIFACTS / "district_catalog.csv").open(encoding="utf-8-sig") as file:
        districts = {row["code"]: row for row in csv.DictReader(file)}
    with (ARTIFACTS / "metrics.json").open(encoding="utf-8") as file:
        scenario_input = json.load(file)["scenario_input"]
    variants = {"base": (1, 1), "double_intensity": (2, 1),
                "double_duration": (1, 2), "double_both": (2, 2)}
    for storm in storms:
        storm["year"] = int(storm["year"])
        storm["month"] = int(storm["month"])
        storm["reported"] = int(storm["reported"])
    oof = []
    scenario_rows = []
    fold_metrics = []
    for fold in range(5):
        training = [s for s in storms if s["year"] <= 2024 and int(s["district_code"]) % 5 != fold]
        test = [s for s in storms if s["year"] == 2025 and int(s["district_code"]) % 5 == fold]
        assert len({s["district_code"] for s in test}) == 10
        assert not ({s["district_code"] for s in training} & {s["district_code"] for s in test})
        assert all(s["year"] <= 2024 for s in training)
        predictions = {}
        for label, use_terrain in (("rain_only", False), ("rain_terrain", True)):
            model = fit(training, use_terrain)
            predictions[label] = model.predict_proba(
                np.array([vector(s, use_terrain) for s in test]))[:, 1]
            if use_terrain:
                terrain_model = model
        train_rain_p99 = np.percentile(np.array([
            [float(s["rain_total_mm"]), float(s["rain_peak_1h_mm"]),
             float(s["rain_duration_h"])] for s in training]), 99, axis=0)
        for code in sorted({s["district_code"] for s in test}):
            district = districts[code]
            scenario_row = {"district_code": code, "district": district["name_en"],
                            "fold": fold, "trained_on_this_district": 0,
                            "base_rate_mm_h": scenario_input["base_rate_mm_h"],
                            "base_duration_h": scenario_input["base_duration_h"],
                            "antecedent_mm": scenario_input["antecedent_mm"],
                            "month": scenario_input["month"]}
            for name, (intensity, duration) in variants.items():
                sample = make_scenario(
                    district,
                    scenario_input["base_rate_mm_h"] * intensity,
                    scenario_input["base_duration_h"] * duration,
                    scenario_input["antecedent_mm"], scenario_input["month"])
                scenario_row[f"p_report_{name}"] = round(float(
                    terrain_model.predict_proba(
                        np.array([vector(sample, True)]))[0, 1]), 6)
                scenario_row[f"outside_training_p99_{name}"] = int(
                    sample["rain_total_mm"] > train_rain_p99[0]
                    or sample["rain_peak_1h_mm"] > train_rain_p99[1]
                    or sample["rain_duration_h"] > train_rain_p99[2])
            scenario_row["delta_double_both"] = round(
                scenario_row["p_report_double_both"] - scenario_row["p_report_base"], 6)
            assert scenario_row["p_report_base"] <= min(
                scenario_row["p_report_double_intensity"],
                scenario_row["p_report_double_duration"]) + 1e-6
            assert scenario_row["p_report_double_both"] + 1e-6 >= max(
                scenario_row["p_report_double_intensity"],
                scenario_row["p_report_double_duration"])
            scenario_rows.append(scenario_row)
        fold_rows = []
        for storm, p_rain, p_terrain in zip(test, predictions["rain_only"], predictions["rain_terrain"]):
            fold_rows.append({"fold": fold, "district_code": storm["district_code"],
                              "district": storm["district_en"],
                              "storm_start": storm["storm_start"],
                              "reported": storm["reported"],
                              "rain_total_mm": storm["rain_total_mm"],
                              "rain_peak_1h_mm": storm["rain_peak_1h_mm"],
                              "rain_duration_h": storm["rain_duration_h"],
                              "p_report_rain_only": round(float(p_rain), 7),
                              "p_report_rain_terrain": round(float(p_terrain), 7)})
        oof.extend(fold_rows)
        by_district = defaultdict(list)
        for row in fold_rows:
            by_district[row["district_code"]].append(row)
        district_rows = list(by_district.values())
        fold_metrics.append({
            "fold": fold, "held_out_districts": [group[0]["district"] for group in district_rows],
            "rain_only_events": event_metrics(test, predictions["rain_only"]),
            "rain_terrain_events": event_metrics(test, predictions["rain_terrain"]),
            "rain_only_district_rank": ranking_agreement(
                [sum(r["p_report_rain_only"] for r in group) for group in district_rows],
                [sum(r["reported"] for r in group) for group in district_rows]),
            "rain_terrain_district_rank": ranking_agreement(
                [sum(r["p_report_rain_terrain"] for r in group) for group in district_rows],
                [sum(r["reported"] for r in group) for group in district_rows]),
        })
    assert len(oof) == sum(s["year"] == 2025 for s in storms)
    assert len(scenario_rows) == 50
    by_district = defaultdict(list)
    for row in oof:
        by_district[row["district_code"]].append(row)
    ranks = []
    for code, group in by_district.items():
        ranks.append({"district_code": code, "district": group[0]["district"],
                      "fold": group[0]["fold"], "storm_count": len(group),
                      "observed_report_storms": sum(r["reported"] for r in group),
                      "predicted_report_storms_rain_only": round(sum(r["p_report_rain_only"] for r in group), 4),
                      "predicted_report_storms_rain_terrain": round(sum(r["p_report_rain_terrain"] for r in group), 4)})
    ranks.sort(key=lambda r: -r["predicted_report_storms_rain_terrain"])
    for i, row in enumerate(ranks, 1):
        row["predicted_rank_rain_terrain"] = i
    rain_scores = np.array([r["p_report_rain_only"] for r in oof])
    terrain_scores = np.array([r["p_report_rain_terrain"] for r in oof])
    summary = {"design": "5 geographic folds; fit on other 40 districts in 2021–2024; test unseen 10 in 2025",
               "test_storms": len(oof), "districts": len(ranks),
               "event_scores": {"rain_only": event_metrics(oof, rain_scores),
                                "rain_terrain": event_metrics(oof, terrain_scores)},
               "district_rank": {
                   "storm_count_baseline": ranking_agreement(
                       [r["storm_count"] for r in ranks],
                       [r["observed_report_storms"] for r in ranks]),
                   "rain_only": ranking_agreement(
                       [r["predicted_report_storms_rain_only"] for r in ranks],
                       [r["observed_report_storms"] for r in ranks]),
                   "rain_terrain": ranking_agreement(
                       [r["predicted_report_storms_rain_terrain"] for r in ranks],
                       [r["observed_report_storms"] for r in ranks])},
               "scenario_input": scenario_input,
               "scenario_target": "Change in model-estimated BMA report occurrence, not physical flood probability",
               "folds": fold_metrics}
    write_csv(ARTIFACTS / "storm_predictions_spatial_oof_2025.csv", oof)
    write_csv(ARTIFACTS / "district_ranking_spatial_oof_2025.csv", ranks)
    scenario_rows.sort(key=lambda r: -r["delta_double_both"])
    write_csv(ARTIFACTS / "scenario_districts_spatial_oof.csv", scenario_rows)
    (ARTIFACTS / "spatial_cv_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))
    if args.district:
        selected = [row for row in scenario_rows
                    if args.district.casefold() in row["district"].casefold()]
        if not selected:
            parser.error(f"No Bangkok district matched {args.district!r}")
        print("Out-of-district scenario:", json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
