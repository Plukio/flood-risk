"""Matched standalone CNN ablations for the unfamiliar-site ranking paper.

Both arms train from scratch on the new-site examples and pairs used by the
site-residual model. Neither receives its fixed nearby-history score. The
physical-only arm zeros the four prior-report input maps during fitting and
scoring; history still chooses matched and hard training comparisons.

Run: .venv/bin/python flood_citywide_site_standalone_ablation.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F

from flood_citywide_history_cnn_transfer import RADIUS, patches
from flood_citywide_history_ranker import (N_DAYS, SHAPE, TEST_START,
                                           TRAIN_START, VALID_START, load_grid)
from flood_citywide_physical_site_data import build
from flood_citywide_spatial_novel_queue import (SpatialNovelCNN, day_maps,
                                                 static_image, top_weights)
from flood_multi_target_citywide_data import OUT
from flood_multi_target_data import START


SOURCE_EVENTS = OUT / "q4_2025_site_novel_rank_curve_events.csv"
RESULT = OUT / "q4_2025_site_standalone_ablation.json"
EVENTS = OUT / "q4_2025_site_standalone_ablation_events.csv"
MODEL = OUT / "q4_2025_site_standalone_ablation.pt"
FRACTIONS = (.01, .02, .05, .10, .20, .30, .50)
MODES = ("cnn_all_inputs", "cnn_physical_only")
EPOCHS = 2
SEED = 30841


def fit(data, source, static, training_events, district, seed, mode):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpatialNovelCNN()
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.02)
    flat = data["candidate_flat"]
    lookup = np.full(data["candidate"].size, -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    # Keep every training day in the shuffle, including days with no new site,
    # so the positive/comparison draws match the original residual arm.
    by_day = defaultdict(lambda: {"new": [], "known": []})
    for event in training_events:
        day = int(event["day_index"])
        if TRAIN_START <= day < VALID_START:
            candidate = int(lookup[event["flat"]])
            if candidate >= 0 and source[candidate]:
                key = "new" if event["new_site"] else "known"
                by_day[day][key].append(candidate)
    losses, pair_counts = [], []
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        pairs = 0
        for day in rng.permutation(list(by_day)):
            image, _, simple, novel = day_maps(data, int(day), static)
            reported = data["reports"][day].ravel()[flat].astype(bool)
            positives = np.unique(np.asarray(by_day[day]["new"], np.int32))
            positives = positives[novel[positives]]
            if not len(positives):
                continue
            positives = rng.choice(positives, size=min(16, len(positives)),
                                   replace=False)
            kept, comparisons = [], []
            for pos in positives:
                pool = np.flatnonzero(source & novel & ~reported &
                                      (district == district[pos]))
                if len(pool) < 2:
                    continue
                order = np.argsort(np.abs(simple[pool]-simple[pos]),
                                   kind="stable")
                matched = rng.choice(pool[order[:min(32, len(pool))]])
                hard = rng.choice(pool[np.argsort(simple[pool])[
                    -max(2, len(pool)//5):]])
                kept.append(int(pos))
                comparisons.extend((int(matched), int(hard)))
            if not kept:
                continue
            pair_pos = np.repeat(np.asarray(kept, np.int32), 2)
            cmp = np.asarray(comparisons, np.int32)
            batch = patches(image, flat, np.r_[pair_pos, cmp])
            if torch.any(batch[:len(pair_pos), :4, RADIUS, RADIUS] != 0):
                raise ValueError("New-site positive exposes own-cell history")
            if mode == "cnn_physical_only":
                batch[:, :4] = 0
            score = model.patch_scores(batch)
            n = len(pair_pos)
            loss = F.softplus(score[n:]-score[:n]).mean()
            loss = loss + .0005*score.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            pairs += n
        losses.append(round(float(np.mean(epoch_losses)), 6))
        pair_counts.append(pairs)
        print("fit", mode, "seed", seed, "epoch", epoch+1,
              "loss", losses[-1], "pairs", pairs, flush=True)
    model.eval()
    return model, {"epoch_pairwise_loss": losses, "pairs_by_epoch": pair_counts}


def percentile(values):
    return rankdata(values, method="average").astype(np.float32)/len(values)


def evaluate(data, model, static, district, members, rows, mode):
    flat = data["candidate_flat"]
    lookup = np.full(data["candidate"].size, -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    ids_by_district = {int(d): np.flatnonzero(district == d) for d in members}
    by_day = defaultdict(list)
    for row in rows:
        day = (date.fromisoformat(row["day"]) - START).days
        by_day[day].append(row)
    physical_score = None
    if mode == "cnn_physical_only":
        image, _, _, _ = day_maps(data, TEST_START, static)
        image[:4] = 0
        physical_score = model.full_scores(image, data["candidate"])
    updated = []
    for day, daily_rows in by_day.items():
        image, _, _, novel = day_maps(data, day, static)
        if physical_score is None:
            score = model.full_scores(image, data["candidate"])
        else:
            score = physical_score
        by_district = defaultdict(list)
        for row in daily_rows:
            by_district[int(row["district_index"])].append(row)
        for d, district_rows in by_district.items():
            ids = ids_by_district[d]
            eligible = ids[novel[ids]]
            value = score[eligible]
            pct = percentile(value)
            local = {int(c): j for j, c in enumerate(eligible)}
            weights = {f: top_weights(value, f*len(value)) for f in FRACTIONS}
            for row in district_rows:
                cell = int(lookup[int(row["row"])*SHAPE[1]+int(row["col"])])
                if cell not in local or len(eligible) != int(row["eligible_cells"]):
                    raise ValueError("Standalone arm changed the candidate pool")
                j = local[cell]
                item = {"fold": int(row["fold"]), "day": row["day"],
                        "district_index": d, "row": int(row["row"]),
                        "col": int(row["col"]),
                        "percentile": float(pct[j])}
                item.update({f"capture_{f:g}": float(weights[f][j])
                             for f in FRACTIONS})
                updated.append(item)
    return updated


def metrics(rows, prefix):
    return {
        "site_events": len(rows),
        "mean_rank_percentile": round(float(np.mean(
            [float(r[f"{prefix}_percentile"]) for r in rows])), 5),
        "capture_curve": [
            {"share_of_unfamiliar_cells": f,
             "expected_sites_found": round(sum(float(r[f"{prefix}_{f:g}"])
                                               for r in rows), 4),
             "site_capture_pct": round(100*sum(float(r[f"{prefix}_{f:g}"])
                                               for r in rows)/len(rows), 3)}
            for f in FRACTIONS
        ],
    }


def main():
    torch.set_num_threads(4)
    with SOURCE_EVENTS.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 393:
        raise ValueError("Expected the frozen 393-site evaluation file")
    data = load_grid()
    training_events = build(data["candidate"], data["district_id"])[250]
    flat = data["candidate_flat"]
    district = data["district_id"].ravel()[flat]
    folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    audit, states = [], {}
    for fold, members in enumerate(folds, 1):
        source = ~np.isin(district, members)
        static, normalization = static_image(data, source)
        fold_rows = [r for r in rows if int(r["fold"]) == fold]
        if set(int(r["district_index"]) for r in fold_rows) - set(members):
            raise ValueError("Frozen events differ from geographic folds")
        states[fold] = {}
        for mode in MODES:
            model, train_audit = fit(data, source, static, training_events,
                                     district, SEED+1000*fold, mode)
            states[fold][mode] = {k: v.cpu() for k, v in model.state_dict().items()}
            predicted = evaluate(data, model, static, district, members,
                                 fold_rows, mode)
            keyed = {(r["day"], r["district_index"], r["row"], r["col"]): r
                     for r in predicted}
            if len(keyed) != len(fold_rows):
                raise ValueError("Standalone evaluation lost or duplicated sites")
            for row in fold_rows:
                key = (row["day"], int(row["district_index"]),
                       int(row["row"]), int(row["col"]))
                found = keyed[key]
                row[f"{mode}_percentile"] = found["percentile"]
                for f in FRACTIONS:
                    row[f"{mode}_{f:g}"] = found[f"capture_{f:g}"]
            audit.append({"fold": fold, "mode": mode, "training": train_audit,
                          "source_normalization": normalization,
                          "q4_eligible_new_sites": len(predicted)})
            print("evaluated", mode, "fold", fold, "sites", len(predicted),
                  flush=True)
        torch.save({"states": states, "audit": audit}, MODEL)
    fieldnames = list(rows[0])
    with EVENTS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "Q4 first-in-year report-site ranking within zero-own-history cells",
        "evaluation": "Same frozen 393 events and per-district/day unfamiliar-cell candidate pools as q4_2025_site_novel_rank_curve.json",
        "training": "2023-24 source-district genuine new-site medoids, 2 epochs, same-district positive-unlabeled pairwise loss, same seed/pairs/architecture",
        "source_selection": "Existing combined model selected on Jan-Sep 2025 source districts; standalone arms have fixed 2-epoch recipes and no Q4 tuning",
        "physical_only_caveat": "Report history is used for training comparison sampling, but four history-map input channels and the fixed history score are absent at inference",
        "arms": {
            "report_history": metrics(rows, "history"),
            "cnn_all_inputs": metrics(rows, "cnn_all_inputs"),
            "cnn_physical_only": metrics(rows, "cnn_physical_only"),
            "history_plus_cnn": metrics(rows, "cnn"),
        },
        "fold_audit": audit,
        "limits": ["Q4 2025 has been repeatedly inspected during development",
                   "Unreported training comparisons are not verified dry",
                   "Site reports are not independently verified flood extent"],
    }
    RESULT.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: {"mean_rank_percentile": v["mean_rank_percentile"],
                          "top_10pct_capture": v["capture_curve"][3]["site_capture_pct"]}
                      for k, v in summary["arms"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
