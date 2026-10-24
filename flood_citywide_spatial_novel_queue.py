"""Dense spatial CNN for first-in-a-year report sites and a fixed-budget queue.

Train only on cells with zero own-cell reports in the previous 365 days.
The CNN sees pre-day report maps, terrain, OSM geometry and WorldCover; its
novel-site score competes only against other zero-history cells. All remaining
inspection slots use the frozen recent+nearby history score. Report absence is
unknown, never a verified dry observation.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label
from torch import nn
from torch.nn import functional as F

from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_multi_target_citywide_data import OUT
from flood_multi_target_citywide_eval import paired_interval
from flood_open_weak_bangkok_holdout_eval import selected
from flood_open_weak_data import FEATURE_NAMES

OUT_JSON = OUT / "q4_2025_spatial_novel_queue_fivefold.json"
OUT_MODEL = OUT / "q4_2025_spatial_novel_queue_fivefold.pt"
STATIC_NAMES = (
    "srtm_elevation_m", "srtm_relative_1250m_m", "srtm_slope",
    "osm_canal_length_log", "osm_canal_distance_log",
    "osm_road_length_log", "osm_road_distance_log",
    "worldcover_built_fraction_2021", "worldcover_crop_fraction_2021",
    "worldcover_tree_grass_fraction_2021",
)
QUOTA_SHARES = (0., .1, .2, .3)
MAX_EPOCHS = 2
MAX_POSITIVES_PER_DAY = 16
MASK_RATE = .10


class SpatialNovelCNN(nn.Module):
    """Depthwise spatial steps compose paths to a 7-cell/1.75 km radius."""

    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(4+len(STATIC_NAMES)+2, 8, 1)
        self.steps = nn.ModuleList([
            nn.ModuleDict({"spatial": nn.Conv2d(8, 8, 3, padding=d,
                                                  dilation=d, groups=8),
                           "mix": nn.Conv2d(8, 8, 1)})
            for d in (1, 1, 1, 2, 2)
        ])
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Last channel is 1 inside the full grid and 0 in patch padding.
        # Masking every layer makes patch scores agree with whole-grid scores
        # even where an extracted patch crosses the grid's outer boundary.
        valid = image[:, -1:]
        features = F.relu(self.entry(image))*valid
        for step in self.steps:
            update = step["mix"](F.relu(step["spatial"](features)))
            features = F.relu(features + .25*update)*valid
        return self.head(features)[:, 0]

    def patch_scores(self, batch: torch.Tensor) -> torch.Tensor:
        return self.forward(batch)[:, RADIUS, RADIUS]

    def full_scores(self, image: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.forward(torch.from_numpy(image[None]))[0].numpy()[candidate]


def static_image(data: dict, source: np.ndarray):
    columns = [FEATURE_NAMES.index(name) for name in STATIC_NAMES]
    raw = data["static"][:, columns].astype(np.float32)
    mean = raw[source].mean(axis=0)
    std = np.maximum(raw[source].std(axis=0), .05)
    normalized = np.clip((raw-mean)/std, -5, 5)
    image = np.zeros((len(columns), *SHAPE), np.float32)
    image[:, data["candidate"]] = normalized.T
    return image, {"names": STATIC_NAMES, "mean": mean.tolist(),
                   "std": std.tolist()}


def day_maps(data: dict, day: int, static: np.ndarray):
    prefix = data["prefix"]
    counts = {k: (prefix[day]-prefix[day-k]).astype(np.float32)
              for k in (365, 90, 30, 7)}
    near = gaussian_filter(counts[365], 1, mode="constant")
    mid = gaussian_filter(counts[365], 3, mode="constant")
    wide = gaussian_filter(counts[365], 6, mode="constant")
    base = np.log1p(counts[365])+.4*np.log1p(counts[90])+.4*np.log1p(near)
    simple = .6*np.log1p(near)+.3*np.log1p(mid)+.1*np.log1p(wide)
    history = np.stack([np.log1p(counts[k]) for k in (365, 90, 30, 7)])
    candidate = data["candidate"]
    image = np.concatenate((history, static, candidate[None].astype(np.float32),
                            np.ones((1, *SHAPE), np.float32))).astype(np.float32)
    return image, base[candidate], simple[candidate], counts[365][candidate] == 0


def top_weights(scores: np.ndarray, capacity: float) -> np.ndarray:
    """Exact fractional top-k allocation, distributing boundary ties evenly."""
    n = len(scores)
    if capacity <= 0 or n == 0:
        return np.zeros(n, np.float32)
    if capacity >= n:
        return np.ones(n, np.float32)
    boundary = np.partition(scores, n-math.ceil(capacity))[n-math.ceil(capacity)]
    above = scores > boundary
    tied = scores == boundary
    share = (capacity-int(above.sum()))/int(tied.sum())
    return above.astype(np.float32)+tied.astype(np.float32)*share


def queue_weights(base: np.ndarray, novelty: np.ndarray, novel: np.ndarray,
                  quota_share: float) -> np.ndarray:
    budget = .1*len(base)
    novelty_budget = quota_share*budget
    known = ~novel
    known_budget = budget-novelty_budget
    if known_budget > known.sum():
        novelty_budget += known_budget-known.sum()
        known_budget = float(known.sum())
    if novelty_budget > novel.sum():
        known_budget += novelty_budget-novel.sum()
        novelty_budget = float(novel.sum())
    result = np.zeros(len(base), np.float32)
    result[known] = top_weights(base[known], known_budget)
    result[novel] = top_weights(novelty[novel], novelty_budget)
    if abs(float(result.sum())-budget) > .01:
        raise ValueError("Inspection budget changed")
    return result


def self_check() -> None:
    rng = np.random.default_rng(SEED)
    for n in (17, 100):
        for scores in (rng.normal(size=n), rng.integers(0, 4, size=n)):
            for fraction in (.05, .1, .2, .37):
                if not np.allclose(top_weights(scores, fraction*n),
                                   selected(scores, fraction), atol=1e-5):
                    raise ValueError("Fractional queue differs from frozen comparator")
    torch.manual_seed(SEED)
    model = SpatialNovelCNN().eval()
    image = rng.normal(size=(4+len(STATIC_NAMES)+2, *SHAPE)).astype(np.float32)
    image[-1] = 1
    flat = np.arange(np.prod(SHAPE))
    chosen = np.array([0, 7*SHAPE[1]+7, 100*SHAPE[1]+100,
                       np.prod(SHAPE)-1])
    with torch.no_grad():
        patch = model.patch_scores(patches(image, flat, chosen)).numpy()
        full = model.forward(torch.from_numpy(image[None]))[0].numpy().ravel()[chosen]
    if not np.allclose(patch, full, atol=1e-5):
        raise ValueError("Novel CNN patch and full-grid scores differ")


def fit(data: dict, source: np.ndarray, static: np.ndarray, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpatialNovelCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    flat = data["candidate_flat"]
    days = []
    for day in range(TRAIN_START, VALID_START):
        own365 = data["prefix"][day].ravel()[flat] == \
                 data["prefix"][day-365].ravel()[flat]
        truth = data["reports"][day].ravel()[flat] > 0
        if np.any(own365 & truth & source):
            days.append(day)
    print("novel-positive training days", len(days), flush=True)
    epoch_losses = []
    for epoch in range(1, MAX_EPOCHS+1):
        losses = []
        model.train()
        for day in rng.permutation(days):
            image, _, simple, novel = day_maps(data, int(day), static)
            truth = data["reports"][day].ravel()[flat] > 0
            positive = np.flatnonzero(novel & truth & source)
            unknown = np.flatnonzero(novel & ~truth & source)
            chosen = rng.choice(positive, size=min(MAX_POSITIVES_PER_DAY,
                                                   len(positive)), replace=False)
            hard_pool = unknown[np.argpartition(simple[unknown],
                max(0, len(unknown)-max(50, len(unknown)//5)))[
                max(0, len(unknown)-max(50, len(unknown)//5)):]]
            random_cmp = rng.choice(unknown, size=len(chosen), replace=True)
            hard_cmp = rng.choice(hard_pool, size=len(chosen), replace=True)
            pos = np.repeat(chosen, 2)
            cmp = np.column_stack((random_cmp, hard_cmp)).ravel()
            indices = np.r_[pos, cmp]
            batch = patches(image, flat, indices)
            if torch.any(batch[:, :4, RADIUS, RADIUS] != 0):
                raise ValueError("Novel training patch exposes own report history")
            if MASK_RATE:
                keep = (torch.rand(len(batch), 1, 2*RADIUS+1,
                                   2*RADIUS+1) >= MASK_RATE).float()
                batch[:, :4] *= keep
            scores = model.patch_scores(batch)
            loss = F.softplus(scores[len(pos):]-scores[:len(pos)]).mean()
            loss = loss + .0005*scores.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_losses.append(round(float(np.mean(losses)),5))
        print("epoch", epoch, "pairwise loss", epoch_losses[-1], flush=True)
    model.eval()
    return model, {"train_days_with_novel_positives": len(days),
                   "epoch_pairwise_loss": epoch_losses,
                   "random_neighbor_history_mask_rate": MASK_RATE}


def collect(data: dict, model: SpatialNovelCNN, static: np.ndarray, days: range):
    bases, simple, learned, novel, truths = [], [], [], [], []
    flat = data["candidate_flat"]
    for day in days:
        image, b, s, n = day_maps(data, day, static)
        predicted = model.full_scores(image, data["candidate"])
        if np.any(~n & (image[0, data["candidate"]] == 0)):
            raise ValueError("Own-history novelty partition changed")
        bases.append(b)
        simple.append(s)
        learned.append(predicted)
        novel.append(n)
        truths.append(data["reports"][day].ravel()[flat].copy())
    return (np.stack(bases), np.stack(simple), np.stack(learned),
            np.stack(novel), np.stack(truths))


def evaluate(data: dict, arrays: tuple[np.ndarray, ...], scope: np.ndarray,
             clusters: bool) -> dict:
    base, simple, learned, novel, truth = arrays
    indices = np.flatnonzero(scope)
    full_flat = data["candidate_flat"][indices]
    names = ["fixed_full"]+[f"{method}_quota_{int(100*q)}pct"
                             for method in ("simple_neighbor", "spatial_cnn")
                             for q in QUOTA_SHARES]
    sums = {name: {"daily_positive": [], "daily_captured": [],
                   "daily_novel_positive": [], "daily_novel_captured": [],
                   "daily_clusters": [], "daily_cluster_hits": []}
            for name in names}
    for day in range(len(truth)):
        b, n, y = base[day, indices], novel[day, indices], truth[day, indices]
        weights = {"fixed_full": top_weights(b, .1*len(indices))}
        for method, score in (("simple_neighbor", simple[day, indices]),
                              ("spatial_cnn", learned[day, indices])):
            for q in QUOTA_SHARES:
                weights[f"{method}_quota_{int(100*q)}pct"] = queue_weights(
                    b, score, n, q)
        if clusters:
            grid = np.zeros(SHAPE, bool)
            grid.ravel()[full_flat] = y > 0
            components, count = label(grid, structure=np.ones((3,3), bool))
            ids = components.ravel()[full_flat]
            members = [ids == j for j in range(1, count+1)]
        for name,w in weights.items():
            row = sums[name]
            row["daily_positive"].append(int(y.sum()))
            row["daily_captured"].append(round(float(np.dot(w,y)),5))
            row["daily_novel_positive"].append(int(y[n].sum()))
            row["daily_novel_captured"].append(round(float(np.dot(w,n*y)),5))
            if clusters:
                row["daily_clusters"].append(count)
                row["daily_cluster_hits"].append(round(
                    sum(float(w[m].max()) for m in members),5))
    out = {}
    for name,row in sums.items():
        positive = sum(row["daily_positive"])
        novel_positive = sum(row["daily_novel_positive"])
        count = sum(row["daily_clusters"])
        out[name] = {"candidate_cells": int(scope.sum()),
                     "positive_cell_days": positive,
                     "top_10pct_capture": round(sum(row["daily_captured"])/positive,5),
                     "novel_positive_cell_days": novel_positive,
                     "novel_capture": round(sum(row["daily_novel_captured"])/novel_positive,5)
                         if novel_positive else None,
                     "same_day_adjacent_clusters": count if clusters else None,
                     "cluster_capture": round(sum(row["daily_cluster_hits"])/count,5)
                         if count else None,
                     **{k:v for k,v in row.items()
                        if k in ("daily_positive", "daily_captured",
                                 "daily_novel_positive", "daily_novel_captured")}}
    return out


def aggregate(folds: list[dict], name: str) -> dict:
    rows = [fold["q4"][name] for fold in folds]
    positive = sum(r["positive_cell_days"] for r in rows)
    novel_positive = sum(r["novel_positive_cell_days"] for r in rows)
    clusters = sum(r["same_day_adjacent_clusters"] for r in rows)
    daily_positive = np.sum([r["daily_positive"] for r in rows], axis=0)
    daily_hits = np.sum([r["daily_captured"] for r in rows], axis=0)
    daily_novel = np.sum([r["daily_novel_positive"] for r in rows], axis=0)
    daily_novel_hits = np.sum([r["daily_novel_captured"] for r in rows], axis=0)
    # Cluster hit totals are reconstructed from rounded fold fractions; the
    # five-decimal precision is sufficient for this exploratory diagnostic.
    cluster_hits = sum(r["cluster_capture"]*r["same_day_adjacent_clusters"]
                       for r in rows)
    return {"candidate_cells": sum(r["candidate_cells"] for r in rows),
            "positive_cell_days": positive,
            "top_10pct_capture": round(float(daily_hits.sum()/positive),5),
            "novel_positive_cell_days": novel_positive,
            "novel_capture": round(float(daily_novel_hits.sum()/novel_positive),5),
            "same_day_adjacent_clusters": clusters,
            "cluster_capture": round(float(cluster_hits/clusters),5),
            "daily_positive": daily_positive.astype(int).tolist(),
            "daily_captured": np.round(daily_hits,5).tolist(),
            "daily_novel_positive": daily_novel.astype(int).tolist(),
            "daily_novel_captured": np.round(daily_novel_hits,5).tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    self_check()
    data = load_grid()
    district = data["district_id"].ravel()[data["candidate_flat"]]
    district_folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    folds, states = [], {}
    for fold_number, ids in enumerate(district_folds, 1):
        target = np.isin(district, ids)
        source = ~target
        print("fold", fold_number, "target cells", target.sum(), flush=True)
        static, normalization = static_image(data, source)
        model, fit_audit = fit(data, source, static, SEED+fold_number*1000)
        states[fold_number] = {k:v.cpu() for k,v in model.state_dict().items()}
        validation = evaluate(data, collect(data, model, static,
                              range(VALID_START, TEST_START)), source, False)
        chosen = {}
        for method in ("simple_neighbor", "spatial_cnn"):
            selected_q = max(QUOTA_SHARES, key=lambda q: validation[
                f"{method}_quota_{int(100*q)}pct"]["top_10pct_capture"])
            chosen[method] = selected_q
        print("fold", fold_number, "source-selected quota shares", chosen,
              flush=True)
        q4 = evaluate(data, collect(data, model, static,
                      range(TEST_START, N_DAYS)), target, True)
        folds.append({"fold": fold_number,
                      "district_codes": [data["districts"][j]["code"] for j in ids],
                      "fit": fit_audit, "normalization": normalization,
                      "source_validation_capture": {k:v["top_10pct_capture"]
                                                   for k,v in validation.items()},
                      "source_selected_quota_share": chosen, "q4": q4})
        print("fold", fold_number,
              {name:(q4[name]["top_10pct_capture"],q4[name]["novel_capture"])
               for name in ("fixed_full", "simple_neighbor_quota_20pct",
                            "spatial_cnn_quota_20pct")}, flush=True)
    names = list(folds[0]["q4"])
    pooled = {name: aggregate(folds, name) for name in names}
    if (pooled["fixed_full"]["candidate_cells"] != 25812 or
        pooled["fixed_full"]["positive_cell_days"] != 7209 or
        pooled["fixed_full"]["novel_positive_cell_days"] != 1664 or
        pooled["fixed_full"]["top_10pct_capture"] != .47113):
        raise ValueError("Frozen matched five-fold queue changed")
    for method in ("simple_neighbor", "spatial_cnn"):
        rows = [f["q4"][f"{method}_quota_{int(100*f['source_selected_quota_share'][method])}pct"]
                for f in folds]
        daily_positive = np.sum([r["daily_positive"] for r in rows],axis=0)
        daily_hits = np.sum([r["daily_captured"] for r in rows],axis=0)
        novel_positive = sum(r["novel_positive_cell_days"] for r in rows)
        novel_hits = sum(sum(r["daily_novel_captured"]) for r in rows)
        cluster_count = sum(r["same_day_adjacent_clusters"] for r in rows)
        cluster_hits = sum(r["cluster_capture"]*r["same_day_adjacent_clusters"]
                           for r in rows)
        pooled[f"{method}_source_selected"] = {
            "positive_cell_days": int(daily_positive.sum()),
            "top_10pct_capture": round(float(daily_hits.sum()/daily_positive.sum()),5),
            "novel_positive_cell_days": novel_positive,
            "novel_capture": round(novel_hits/novel_positive,5),
            "same_day_adjacent_clusters": cluster_count,
            "cluster_capture": round(cluster_hits/cluster_count,5),
            "daily_positive": daily_positive.astype(int).tolist(),
            "daily_captured": np.round(daily_hits,5).tolist()}
        pooled[f"{method}_source_selected_minus_fixed_weekly_interval"] = paired_interval(
            pooled[f"{method}_source_selected"],pooled["fixed_full"])
    output = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "target": "Traffy report-positive 250m cell-days; not verified flooding",
              "training_period": ["2023-01-01","2024-12-31"],
              "source_validation_period": ["2025-01-01","2025-09-30"],
              "heldout_test_period": ["2025-10-01","2025-12-31"],
              "split": "same five district folds and 10% daily inspection budget as earlier CNN",
              "quota_shares_of_total_budget": QUOTA_SHARES,
              "simple_novel_score": "0.6 log1p(250m-smoothed 365d count) + 0.3 log1p(750m-smoothed) + 0.1 log1p(1500m-smoothed)",
              "cnn": "five spatial steps with 3x3 kernels and dilation 1,1,1,2,2; source-normalized terrain/OSM/cover; own history is exactly zero for novel candidates",
              "training_loss": "same-day positive-versus-unreported pairwise loss within zero-own-365d source candidates; random and hard neighbor-history comparisons",
              "random_neighbor_history_mask_rate": MASK_RATE,
              "folds": folds, "pooled_fold_queues": pooled,
              "limitations": ["Unreported candidate cells are unknown, not dry",
                              "No-own-history is not necessarily a new physical site",
                              "Q4 2025 has been inspected in previous work",
                              "SRTM, OSM and WorldCover snapshots may differ from event dates",
                              "No verified water flow or flood extent; no rain simulation"]}
    OUT_JSON.write_text(json.dumps(output,indent=2)+"\n")
    torch.save({"states":states},OUT_MODEL)
    print("pooled",{k:v["top_10pct_capture"] for k,v in pooled.items()
                    if isinstance(v,dict) and "top_10pct_capture" in v},flush=True)
    print("saved",OUT_JSON,OUT_MODEL,flush=True)


if __name__ == "__main__":
    main()
