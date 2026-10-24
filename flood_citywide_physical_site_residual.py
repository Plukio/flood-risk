"""Train a site-label CNN residual over nearby report-history ranking.

The positive training target is a direct-category Traffy point-cluster medoid
with no report within 250 m on an earlier day in the previous 365 days.
Both training arms use a same-district, similar-history pairwise loss. The
second arm adds repeat-site positives with a whole 3x3 historical-report
neighborhood hidden, testing whether synthetic cold starts help transfer.
Unreported comparison cells remain unlabeled, not verified dry.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from flood_citywide_history_cnn_transfer import RADIUS, patches
from flood_citywide_history_ranker import N_DAYS, TEST_START, TRAIN_START, VALID_START, load_grid
from flood_citywide_physical_site_data import build
from flood_citywide_physical_site_replay import (BLENDS, baseline_capture,
                                                 event_capture, group_events)
from flood_citywide_spatial_novel_queue import (SpatialNovelCNN, collect,
                                                 day_maps, static_image)
from flood_multi_target_citywide_data import OUT


RESULT = OUT / "q4_2025_physical_site_residual_fivefold.json"
MODEL = OUT / "q4_2025_physical_site_residual_fivefold.pt"
ARMS = ("site_new_only", "site_new_plus_group_mask")
EPOCHS = 2
SEED = 30841
RADIUS_M = 250


def fit(data, source, static, training_events, district, seed, synthetic):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpatialNovelCNN()
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.02)
    flat = data["candidate_flat"]
    lookup = np.full(data["candidate"].size, -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    by_day = defaultdict(lambda: {"new": [], "known": []})
    for event in training_events:
        day = event["day_index"]
        if not TRAIN_START <= day < VALID_START:
            continue
        candidate = int(lookup[event["flat"]])
        if candidate < 0 or not source[candidate]:
            continue
        key = "new" if event["new_site"] else "known"
        by_day[day][key].append(candidate)
    epoch_loss, counts = [], {"new": 0, "synthetic": 0, "pairs": 0}
    for epoch in range(EPOCHS):
        losses = []
        for day in rng.permutation(list(by_day)):
            image, _, simple, novel = day_maps(data, int(day), static)
            reported = data["reports"][day].ravel()[flat].astype(bool)
            true_new = np.unique(np.asarray(by_day[day]["new"], np.int32))
            true_new = true_new[novel[true_new]]
            if not len(true_new):
                continue
            true_new = rng.choice(true_new, size=min(16, len(true_new)), replace=False)
            old = (np.unique(np.asarray(by_day[day]["known"], np.int32))
                   if synthetic else np.array([], np.int32))
            old = old[novel[old] == False] if len(old) else old
            if len(old):
                old = rng.choice(old, size=min(8, len(old), len(true_new)), replace=False)
            positives = np.r_[true_new, old]
            comparisons = []
            kept = []
            for i, pos in enumerate(positives):
                pool = np.flatnonzero(source & novel & ~reported &
                                      (district == district[pos]))
                if len(pool) < 2:
                    continue
                order = np.argsort(np.abs(simple[pool]-simple[pos]), kind="stable")
                matched = rng.choice(pool[order[:min(32, len(pool))]])
                hard = rng.choice(pool[np.argsort(simple[pool])[-max(2, len(pool)//5):]])
                kept.append((int(pos), i >= len(true_new)))
                comparisons.extend((int(matched), int(hard)))
            if not kept:
                continue
            pos = np.asarray([x[0] for x in kept], np.int32)
            cmp = np.asarray(comparisons, np.int32)
            # Both comparisons see the same positive input, including its mask.
            pair_pos = np.repeat(pos, 2)
            batch = patches(image, flat, np.r_[pair_pos, cmp])
            for j, (_, is_synthetic) in enumerate(kept):
                if is_synthetic:
                    batch[2*j:2*j+2, :4, RADIUS-1:RADIUS+2,
                          RADIUS-1:RADIUS+2] = 0
            if torch.any(batch[:2*len(pos), :4, RADIUS, RADIUS] != 0):
                raise ValueError("Positive own-cell history leaked through site mask")
            # The model learns a residual, initially zero, over the nearby
            # history score. Comparison cells are positive-unlabeled rivals.
            residual = model.patch_scores(batch)
            fixed = torch.from_numpy(simple[np.r_[pair_pos, cmp]].copy())
            score = fixed + residual
            n = len(pair_pos)
            per_pair = F.softplus(score[n:]-score[:n])
            weight = torch.tensor([.3 if kept[j//2][1] else 1.
                                   for j in range(n)])
            loss = (per_pair*weight).sum()/weight.sum()
            loss = loss+.0005*residual.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            counts["new"] += sum(not x[1] for x in kept)
            counts["synthetic"] += sum(x[1] for x in kept)
            counts["pairs"] += n
        epoch_loss.append(round(float(np.mean(losses)), 6))
        print("fit", "group-mask" if synthetic else "new-only", "epoch", epoch+1,
              "loss", epoch_loss[-1], "examples", counts, flush=True)
    model.eval()
    return model, {"epoch_pairwise_loss": epoch_loss, "samples": counts,
                   "synthetic_repeat_mask": "3x3 history cells, weight 0.3" if synthetic else "none"}


def score_days(data, model, static, days):
    arrays = collect(data, model, static, days)
    # The learned output is a correction to the fixed nearby score.
    score = arrays[1]+arrays[2]
    return arrays, score


def select_blend(arrays, score, grouped, district, source_members):
    rows = {}
    for alpha in BLENDS:
        result, _ = event_capture(arrays, score, grouped, district,
                                  source_members, RADIUS_M, alpha)
        rows[str(alpha)] = result["new_site"]
    best = max(BLENDS, key=lambda a: (rows[str(a)]["model_capture_pct"], -a))
    return best, rows


def main():
    torch.set_num_threads(4)
    data = load_grid()
    events = build(data["candidate"], data["district_id"])[RADIUS_M]
    flat = data["candidate_flat"]
    district = data["district_id"].ravel()[flat]
    lookup = np.full(data["candidate"].size, -1, np.int32)
    lookup[flat] = np.arange(len(flat))
    folds = np.array_split(np.random.default_rng(917).permutation(len(data["districts"])), 5)
    output, weights = [], {}
    for fold, members in enumerate(folds, 1):
        target = np.isin(district, members)
        source = ~target
        source_members = np.setdiff1d(np.arange(len(data["districts"])), members)
        static, _ = static_image(data, source)
        source_allowed = np.zeros(len(data["districts"]), bool)
        source_allowed[source_members] = True
        target_allowed = ~source_allowed
        val_group = group_events(events, source_allowed, VALID_START, TEST_START, lookup)
        q4_group = group_events(events, target_allowed, TEST_START, N_DAYS, lookup)
        arms = {}
        weights[fold] = {}
        for arm in ARMS:
            model, train_audit = fit(data, source, static, events, district,
                                     SEED+1000*fold, arm.endswith("group_mask"))
            weights[fold][arm] = {k: v.cpu() for k, v in model.state_dict().items()}
            val_arrays, val_score = score_days(data, model, static,
                                                range(VALID_START, TEST_START))
            alpha, validation = select_blend(val_arrays, val_score, val_group,
                                              district, source_members)
            del val_arrays, val_score
            q4_arrays, q4_score = score_days(data, model, static,
                                             range(TEST_START, N_DAYS))
            result, _ = event_capture(q4_arrays, q4_score, q4_group, district,
                                      members, RADIUS_M, alpha)
            baseline = baseline_capture(q4_arrays, q4_group, district, members)
            arms[arm] = {"fit": train_audit, "source_validation_new_site": validation,
                         "source_selected_alpha": alpha,
                         "q4_baseline": baseline["new_site"],
                         "q4_model": result["new_site"]}
            print("fold", fold, arm, "alpha", alpha,
                  "Q4", result["new_site"], flush=True)
        output.append({"fold": fold,
                       "district_codes": [data["districts"][i]["code"] for i in members],
                       "arms": arms})
    pooled = {}
    for arm in ARMS:
        rows = [f["arms"][arm] for f in output]
        n = sum(x["q4_model"]["events"] for x in rows)
        base = sum(x["q4_baseline"]["hits"] for x in rows)
        model = sum(x["q4_model"]["model_hits"] for x in rows)
        pooled[arm] = {"new_site_events": n,
                       "baseline_hits": round(base, 4),
                       "model_hits": round(model, 4),
                       "baseline_capture_pct": round(100*base/n, 3),
                       "model_capture_pct": round(100*model/n, 3)}
    result = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "site_radius_m": RADIUS_M,
              "training": "2023-24 source-district physical-new site medoids, pairwise vs same-district unreported zero-history cells",
              "synthetic_arm": "Repeat-site medoids additionally masked in a 3x3 history patch at 0.3 loss weight",
              "selection": "Jan-Sep 2025 source-district site-event capture chooses blend with nearby history",
              "test": "Q4 2025 excluded districts; same 10% daily district cells, 20% novel reserve",
              "folds": output, "pooled_q4": pooled,
              "limits": ["Q4 2025 has been repeatedly inspected",
                         "No report is not verified dry ground",
                         "A report-site medoid is not flood extent"]}
    RESULT.write_text(json.dumps(result, indent=2)+"\n")
    torch.save({"states": weights}, MODEL)
    print("pooled", json.dumps(pooled, indent=2), flush=True)


if __name__ == "__main__":
    main()
