"""Aggregate three matched seeds of the joint target-site mask holdouts."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from flood_joint_target_site_mask_train import DEST

SEEDS=(18319,18320,18321)
OUT=DEST/"three_seed_summary.json"


def main() -> None:
    rows=[]
    for seed in SEEDS:
        directory=DEST if seed==SEEDS[0] else DEST/f"seed_{seed}"
        training=json.loads((directory/"training_summary.json").read_text())
        evaluation=json.loads((directory/"holdout_evaluation.json").read_text())
        if training["seed"]!=seed or len(training["training"])!=4:
            raise ValueError(f"Incomplete matched four-arm seed {seed}")
        draws=[r["task_counts"] for r in training["training"]]
        if len({(d["traffy"],d["bma"],d["actual_first_in_year"],
                 d["synthetic_repeat"]) for d in draws})!=1:
            raise ValueError(f"Urban task draws differ within seed {seed}")
        urban=evaluation["urban"]["outside_urban_training_patches"]
        if (urban["target_centered_3x3"]["isolated_positive_cell_days"]!=220
            or urban["target_centered_3x3"]["novel_positive_cell_days"]!=959):
            raise ValueError(f"Outside-patch denominators changed for seed {seed}")
        satellite=evaluation["satellite_positive_area"]["groups"]
        bma=evaluation["bma_positive_sites"]["groups"]
        rows.append({"seed":seed,
                     "urban_outside_patches":{name:{
                         "overall_captured":urban[name]["captured"],
                         "novel_captured":urban[name]["novel_captured"],
                         "isolated_captured":urban[name]["isolated_captured"]}
                         for name in ("random_block_matched",
                                      "target_centered_3x3",
                                      "urban_only_random_block_matched",
                                      "urban_only_target_centered_3x3")},
                     "bma_isolated_site_days_captured":{
                         name:bma["novel_isolated"]["captured"][name]
                         for name in ("random_block_matched",
                                      "target_centered_3x3")},
                     "satellite_top_10pct_area_capture":{
                         key:satellite[key]["top_10pct_capture"]
                         for key in ("pathum_thani_gistda_8_dates",
                                     "bangkok_eos_2_patches",
                                     "pathum_thani_unosat_2_wet_dates")}})
    differences={
        "joint_center_minus_joint_random_isolated_cells":[
            r["urban_outside_patches"]["target_centered_3x3"]["isolated_captured"]-
            r["urban_outside_patches"]["random_block_matched"]["isolated_captured"]
            for r in rows],
        "joint_center_minus_joint_random_all_novel_cells":[
            r["urban_outside_patches"]["target_centered_3x3"]["novel_captured"]-
            r["urban_outside_patches"]["random_block_matched"]["novel_captured"]
            for r in rows],
        "joint_center_minus_urban_only_center_isolated_cells":[
            r["urban_outside_patches"]["target_centered_3x3"]["isolated_captured"]-
            r["urban_outside_patches"]["urban_only_target_centered_3x3"]["isolated_captured"]
            for r in rows],
        "joint_random_minus_urban_only_random_all_novel_cells":[
            r["urban_outside_patches"]["random_block_matched"]["novel_captured"]-
            r["urban_outside_patches"]["urban_only_random_block_matched"]["novel_captured"]
            for r in rows]}
    if differences["joint_center_minus_joint_random_isolated_cells"]!=[14.,1.,0.]:
        raise ValueError("Frozen isolated mask comparison changed")
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seeds":SEEDS,"rows":rows,"paired_seed_differences":differences,
            "interpretation":"Same Q4 and satellite holdouts inspected for all seeds; seed repeats probe initialization and sampling sensitivity, not independent test sets"}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(differences,indent=2))
    print("saved",OUT)


if __name__=="__main__":
    main()
