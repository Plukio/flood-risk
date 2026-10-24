"""Matched strict-isolated queue: target-history masking with/without path pretraining.

Reuses the saved, geographically separated D8 matched-path pretraining states.
Both report arms use identical LiDAR features, masked examples, group loss,
source-only alpha selection, and held-out district Q4 queue.

Run: .venv/bin/python flood_citywide_path_pretrain_mask_transfer.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_citywide_history_cnn_transfer import SEED
from flood_citywide_history_ranker import N_DAYS, TEST_START, VALID_START, load_grid
from flood_citywide_isolated_context_residual import (
    ALPHAS, ContextResidualCNN, pool, score_days,
)
from flood_citywide_isolated_group_mask import (
    mask_self_check, train, transfer_path_spatial_weights,
)
from flood_citywide_path_contrast_pretrain import OUT_MODEL as PATH_MODEL
from flood_citywide_route_pretrain import OUT, lidar_static, terrain
from flood_citywide_spatial_novel_queue import SpatialNovelCNN

ARM = "isolated_plus_masked_adjacent"
OUT_JSON = OUT / "q4_2025_path_pretrain_target_mask_matched_fivefold.json"
OUT_MODEL = OUT / "q4_2025_path_pretrain_target_mask_matched_fivefold.pt"
PRETRAIN_MODEL = PATH_MODEL.with_name(PATH_MODEL.stem+"_reset_head.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_grid()
    elevation, _, _, terrain_audit = terrain(data)
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    saved = torch.load(PRETRAIN_MODEL, map_location="cpu", weights_only=True)
    records, states = [], {}
    for fold_id, ids in enumerate(folds, 1):
        if args.fold is not None and fold_id != args.fold:
            continue
        source = ~np.isin(district, ids)
        static, norm = lidar_static(data, source, elevation)
        mask_self_check(data, static)
        pre = SpatialNovelCNN()
        pre.load_state_dict(saved["states"][fold_id]["synthetic"])
        # Fail early if any transferred tensor or channel mapping changes.
        check = ContextResidualCNN()
        transfer_path_spatial_weights(check, pre)
        assert torch.equal(check.entry.weight[:, :1], pre.entry.weight[:, :1])
        assert torch.equal(check.entry.weight[:, 4:15], pre.entry.weight[:, 4:15])
        assert torch.equal(check.entry.weight[:, 20], pre.entry.weight[:, 15])
        assert torch.count_nonzero(check.entry.weight[:, 1:4]) == 0
        assert torch.count_nonzero(check.entry.weight[:, 15:20]) == 0
        assert torch.count_nonzero(check.head.weight) == 0
        print("fold", fold_id, "source", int(source.sum()), flush=True)
        record = {"fold": fold_id,
                  "heldout_district_codes": [data["districts"][j]["code"]
                                             for j in ids],
                  "normalization": norm, "arms": {}}
        states[fold_id] = {}
        for name, initial in (("scratch", None), ("path_pretrained", pre)):
            model, fit = train(data, source, static,
                               SEED+fold_id*1000+args.seed_offset,
                               ARM, pretrained=initial)
            valid = score_days(data, source, static, model,
                               range(VALID_START, TEST_START))
            alpha = max(ALPHAS, key=lambda a: (
                valid[str(a)]["isolated_group_capture"],
                valid[str(a)]["isolated_cell_capture"], -a))
            test = score_days(data, ~source, static, model,
                              range(TEST_START, N_DAYS))
            record["arms"][name] = {
                "fit": fit, "source_validation": valid,
                "selected_alpha": alpha, "heldout_q4": test,
            }
            states[fold_id][name] = {k: v.cpu() for k, v in model.state_dict().items()}
            print("fold", fold_id, name, "alpha", alpha,
                  "q4 cells", test[str(alpha)]["isolated_hits"],
                  "groups", test[str(alpha)]["group_hits"], flush=True)
        records.append(record)
    pooled = {}
    for name in ("scratch", "path_pretrained"):
        pooled[name] = {
            str(a): pool([f["arms"][name]["heldout_q4"][str(a)]
                          for f in records]) for a in ALPHAS
        }
        pooled[name]["source_selected"] = pool([
            f["arms"][name]["heldout_q4"][str(f["arms"][name]["selected_alpha"])]
            for f in records])
        if args.fold is None:
            baseline = pooled[name]["0.0"]
            if (baseline["all_positive"] != 7209 or
                baseline["isolated_positive"] != 332 or
                baseline["isolated_groups"] != 158 or
                baseline["isolated_hits"] != 65. or
                baseline["group_hits"] != 44.):
                raise ValueError("Frozen strict-isolated queue changed")
    result = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": "Does D8 matched-path pretraining improve the correctly target-masked group CNN?",
        "report_finetuning_seed_offset": args.seed_offset,
        "target": "Traffy report-positive 250m cell-days, not verified flooding",
        "report_training_arm": ARM,
        "synthetic_pretraining_checkpoint": str(PRETRAIN_MODEL),
        "transfer": "Copy source/365-day history, LiDAR/static/candidate/valid entry weights and five spatial steps; zero other history and five wider-history weights; reset report head",
        "features": "Same source-normalized LiDAR, roads, canals, 2021 WorldCover, four pre-day report histories, and five wider Gaussian report histories in both arms",
        "loss": "Same masked positive-group logmeanexp pairwise ranking against random and hard unreported comparisons in both arms",
        "mask": "Exact target 3x3 removal from raw report history and five Gaussian layers for positives and comparisons",
        "dates": {"report_training": ["2023-01-01", "2024-12-31"],
                  "source_validation": ["2025-01-01", "2025-09-30"],
                  "heldout_q4": ["2025-10-01", "2025-12-31"]},
        "queue": "10% of heldout-fold cells per day, 20% of slots reserved for strict isolated candidates",
        "alpha_selection": "Best source-validation isolated group capture, then cell capture, then smaller alpha",
        "terrain": terrain_audit,
        "folds": records,
        "pooled_heldout_q4": pooled,
        "limitations": ["Unreported comparisons are unknown, not verified dry",
                        "D8 connections are terrain proxies, not observed drains",
                        "Q4 2025 has been inspected repeatedly; comparison is exploratory",
                        "Present-day LiDAR and static layers are not vintage-matched",
                        "No verified water extent or rainfall simulation validity"],
    }
    suffix = ("" if args.seed_offset == 0 else f"_seed_{args.seed_offset}")
    if args.fold is not None:
        suffix += f"_fold_{args.fold}"
    out_json = OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    out_model = OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    out_json.write_text(json.dumps(result, indent=2)+"\n")
    torch.save({"states": states}, out_model)
    print("pooled", {name: {"cells": rows["source_selected"]["isolated_hits"],
                            "groups": rows["source_selected"]["group_hits"]}
                     for name, rows in pooled.items()}, flush=True)
    print("saved", out_json, out_model, flush=True)


if __name__ == "__main__":
    main()
