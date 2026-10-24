"""Geographic holdout for genuinely isolated Traffy report sites.

An eligible cell has no report in its own or eight neighboring 250 m cells
during the 365 days before scoring. Positives are grouped by same-day touching
cells before training so a complaint buffer does not supply multiple examples.
Unreported comparison cells are unknown, not observed dry.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label, maximum_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.nn import functional as F

from flood_citywide_history_ranker import (
    N_DAYS, SHAPE, TEST_START, TRAIN_START, VALID_START, load_grid,
)
from flood_citywide_history_cnn_transfer import RADIUS, SEED, patches
from flood_citywide_spatial_novel_queue import (
    SpatialNovelCNN, day_maps, static_image, top_weights,
)
from flood_multi_target_citywide_data import OUT
from flood_open_weak_data import FEATURE_NAMES

OUT_JSON = OUT / "q4_2025_isolated_site_ranker_fivefold.json"
OUT_MODEL = OUT / "q4_2025_isolated_site_ranker_fivefold.pt"
RESERVE = .20  # Of a 10% daily inspection budget; fixed before Q4 scoring.
EPOCHS = 2
STATIC_COLUMNS = tuple(FEATURE_NAMES.index(n) for n in (
    "srtm_elevation_m", "srtm_relative_1250m_m", "srtm_slope",
    "osm_canal_length_log", "osm_canal_distance_log",
    "osm_road_length_log", "osm_road_distance_log",
    "worldcover_built_fraction_2021", "worldcover_crop_fraction_2021",
    "worldcover_tree_grass_fraction_2021",
))


def isolated(data: dict, day: int) -> np.ndarray:
    prior = (data["prefix"][day] - data["prefix"][day-365]) > 0
    return data["candidate"] & ~maximum_filter(
        prior.astype(np.uint8), size=3, mode="constant").astype(bool)


def day_inputs(data: dict, day: int, static: np.ndarray,
               include_distant_layers: bool = False):
    image, base, _, _ = day_maps(data, day, static)
    prior365 = (data["prefix"][day] - data["prefix"][day-365]).astype(np.float32)
    prior90 = (data["prefix"][day] - data["prefix"][day-90]).astype(np.float32)
    # All eligible cells have zero history in their 3x3 neighborhood. These
    # wider smooths measure strictly pre-day reports farther away.
    distant = [gaussian_filter(prior365, s, mode="constant")
               for s in (3, 6, 12)]
    distant += [gaussian_filter(prior90, s, mode="constant")
                for s in (3, 6)]
    mask = data["candidate"]
    x = np.column_stack((data["static"][:, STATIC_COLUMNS],
                         *[np.log1p(z[mask]) for z in distant])).astype(np.float32)
    simple = (.6*np.log1p(distant[0][mask]) +
              .3*np.log1p(distant[1][mask]) +
              .1*np.log1p(distant[2][mask])).astype(np.float32)
    if include_distant_layers:
        return image, base, simple, x, np.stack(
            [np.log1p(z) for z in distant]).astype(np.float32)
    return image, base, simple, x


def grouped_representatives(positive: np.ndarray, rng: np.random.Generator,
                            limit: int | None = None) -> np.ndarray:
    components, count = label(positive, structure=np.ones((3, 3), bool))
    if not count:
        return np.empty(0, np.int32)
    ids = np.arange(1, count+1)
    if limit is not None and count > limit:
        ids = rng.choice(ids, size=limit, replace=False)
    flattened = components.ravel()
    return np.array([rng.choice(np.flatnonzero(flattened == j)) for j in ids],
                    dtype=np.int32)


def samples(data: dict, day: int, source_grid: np.ndarray,
            simple: np.ndarray, rng: np.random.Generator,
            limit: int | None = None):
    eligibility = isolated(data, day)
    truth = data["reports"][day] > 0
    pos = grouped_representatives(eligibility & truth & source_grid, rng, limit)
    if not len(pos):
        return None
    # Only source-district outcomes may affect training, including the
    # comparison exclusion buffer near district borders.
    same_day_source = truth & source_grid
    safe = (eligibility & source_grid & ~maximum_filter(
        same_day_source.astype(np.uint8), size=3, mode="constant").astype(bool))
    unknown = np.flatnonzero(safe.ravel())
    if not len(unknown):
        return None
    flat_to_index = np.full(np.prod(SHAPE), -1, np.int32)
    flat_to_index[data["candidate_flat"]] = np.arange(len(data["candidate_flat"]))
    unknown_idx = flat_to_index[unknown]
    hard_n = min(len(unknown), max(50, len(unknown)//5))
    hard = unknown[np.argpartition(simple[unknown_idx], -hard_n)[-hard_n:]]
    neg = np.column_stack((rng.choice(unknown, len(pos), replace=True),
                           rng.choice(hard, len(pos), replace=True))).ravel()
    return flat_to_index[np.repeat(pos, 2)], flat_to_index[neg], len(pos)


def train_tabular(data: dict, source: np.ndarray, static: np.ndarray,
                  seed: int):
    rng = np.random.default_rng(seed)
    source_grid = np.zeros(SHAPE, bool)
    source_grid.ravel()[data["candidate_flat"]] = source
    xs, ys, groups = [], [], 0
    for day in range(TRAIN_START, VALID_START):
        if not np.any((data["reports"][day] > 0) &
                      isolated(data, day) & source_grid):
            continue
        _, _, simple, x = day_inputs(data, day, static)
        picked = samples(data, day, source_grid, simple, rng)
        if picked is None:
            continue
        pos, neg, n_groups = picked
        xs.extend((x[pos], x[neg]))
        ys.extend((np.ones(len(pos), np.uint8), np.zeros(len(neg), np.uint8)))
        groups += n_groups
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=9,
        min_samples_leaf=25, learning_rate=.05, l2_regularization=3.,
        random_state=seed)
    model.fit(np.concatenate(xs), np.concatenate(ys))
    return model, {"positive_site_groups_sampled":groups,
                   "positive_training_rows":groups*2,
                   "unreported_comparison_rows":groups*2}


def train_cnn(data: dict, source: np.ndarray, static: np.ndarray, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    source_grid = np.zeros(SHAPE, bool)
    source_grid.ravel()[data["candidate_flat"]] = source
    days = [d for d in range(TRAIN_START, VALID_START)
            if np.any((data["reports"][d] > 0) &
                      isolated(data, d) & source_grid)]
    model = SpatialNovelCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    losses, used_groups = [], 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for day in rng.permutation(days):
            image, _, simple, _ = day_inputs(data, int(day), static)
            picked = samples(data, int(day), source_grid, simple, rng, limit=12)
            if picked is None:
                continue
            pos, neg, n_groups = picked
            used_groups += n_groups
            batch = patches(image, data["candidate_flat"], np.r_[pos, neg])
            if torch.any(batch[:, :4, RADIUS-1:RADIUS+2,
                                      RADIUS-1:RADIUS+2] != 0):
                raise ValueError("Isolated candidate exposes own/adjacent history")
            score = model.patch_scores(batch)
            loss = F.softplus(score[len(pos):]-score[:len(pos)]).mean()
            loss = loss + .0005*score.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
        print("CNN epoch",epoch+1,"loss",round(losses[-1],5),flush=True)
    model.eval()
    return model, {"positive_site_groups_sampled_across_epochs":used_groups,
                   "epoch_pairwise_loss":losses}


def queue(base: np.ndarray, score: np.ndarray, eligible: np.ndarray):
    budget = .1*len(base)
    reserved = RESERVE*budget
    if reserved > eligible.sum():
        reserved = float(eligible.sum())
    out = np.zeros(len(base), np.float32)
    out[eligible] = top_weights(score[eligible], reserved)
    out[~eligible] = top_weights(base[~eligible], budget-reserved)
    if abs(float(out.sum())-budget) > .02:
        raise ValueError("Queue budget drift")
    return out


def evaluate(data: dict, source: np.ndarray, static: np.ndarray,
             tabular, cnn, days: range):
    candidates = data["candidate"]
    flat = data["candidate_flat"]
    scoped = np.flatnonzero(source)
    scoped_flat = flat[scoped]
    method_names = ("history_no_reserve", "distant_history", "tabular", "cnn")
    rows = {name:{"all_positive":0,"all_hits":0.,"isolated_positive":0,
                  "isolated_hits":0.,"isolated_groups":0,"group_hits":0.,
                  "daily_isolated_positive":[],"daily_isolated_hits":[]}
            for name in method_names}
    for day in days:
        image, base, simple, x = day_inputs(data, day, static)
        elig = isolated(data, day)[candidates]
        truth = data["reports"][day].ravel()[flat] > 0
        score_hgb = tabular.predict_proba(x)[:,1].astype(np.float32)
        score_cnn = cnn.full_scores(image, candidates)
        e = elig[scoped]
        y = truth[scoped]
        iso_y = y & e
        components, n_groups = label(
            np.reshape(np.bincount(scoped_flat, weights=iso_y,
                                   minlength=np.prod(SHAPE)), SHAPE) > 0,
            structure=np.ones((3,3),bool))
        component_ids = components.ravel()[scoped_flat]
        picks = {"history_no_reserve":top_weights(base[scoped], .1*len(scoped)),
                 "distant_history":queue(base[scoped], simple[scoped], e),
                 "tabular":queue(base[scoped], score_hgb[scoped], e),
                 "cnn":queue(base[scoped], score_cnn[scoped], e)}
        for name,w in picks.items():
            r=rows[name]
            r["all_positive"] += int(y.sum())
            r["all_hits"] += float(w[y].sum())
            r["isolated_positive"] += int(iso_y.sum())
            hits=float(w[iso_y].sum())
            r["isolated_hits"] += hits
            r["isolated_groups"] += n_groups
            r["group_hits"] += sum(float(w[component_ids==j].max())
                                     for j in range(1,n_groups+1))
            r["daily_isolated_positive"].append(int(iso_y.sum()))
            r["daily_isolated_hits"].append(round(hits,5))
    for r in rows.values():
        r["all_capture"] = round(r["all_hits"]/r["all_positive"],5)
        r["isolated_cell_capture"] = round(
            r["isolated_hits"]/r["isolated_positive"],5)
        r["isolated_group_capture"] = round(
            r["group_hits"]/r["isolated_groups"],5)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    suffix = "" if args.seed_offset == 0 else f"_seed_{args.seed_offset}"
    out_json = OUT_JSON if not suffix else OUT_JSON.with_name(OUT_JSON.stem+suffix+".json")
    out_model = OUT_MODEL if not suffix else OUT_MODEL.with_name(OUT_MODEL.stem+suffix+".pt")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_grid()
    district = data["district_id"].ravel()[data["candidate_flat"]]
    folds = np.array_split(np.random.default_rng(917).permutation(
        len(data["districts"])), 5)
    records, states = [], {}
    for fold_id, ids in enumerate(folds, 1):
        source = ~np.isin(district, ids)
        print("fold",fold_id,"source",int(source.sum()),flush=True)
        static, norm = static_image(data, source)
        seed = SEED+fold_id*1000+args.seed_offset
        tabular, tab_fit = train_tabular(data, source, static, seed)
        cnn, cnn_fit = train_cnn(data, source, static, seed)
        validation = evaluate(data, source, static, tabular, cnn,
                              range(VALID_START, TEST_START))
        test = evaluate(data, ~source, static, tabular, cnn,
                        range(TEST_START, N_DAYS))
        records.append({"fold":fold_id,
                        "heldout_district_codes":[data["districts"][j]["code"]
                                                  for j in ids],
                        "normalization":norm,"tabular_fit":tab_fit,
                        "cnn_fit":cnn_fit,"source_validation":validation,
                        "heldout_q4":test})
        states[fold_id]={k:v.cpu() for k,v in cnn.state_dict().items()}
        print("fold",fold_id,{k:(v["isolated_cell_capture"],
                                    v["isolated_group_capture"])
                              for k,v in test.items()},flush=True)
    pooled={}
    for name in records[0]["heldout_q4"]:
        rs=[fold["heldout_q4"][name] for fold in records]
        keys=("all_positive","all_hits","isolated_positive","isolated_hits",
              "isolated_groups","group_hits")
        row={k:sum(r[k] for r in rs) for k in keys}
        row["all_capture"]=round(row["all_hits"]/row["all_positive"],5)
        row["isolated_cell_capture"]=round(
            row["isolated_hits"]/row["isolated_positive"],5)
        row["isolated_group_capture"]=round(
            row["group_hits"]/row["isolated_groups"],5)
        pooled[name]=row
    if (pooled["history_no_reserve"]["all_positive"] != 7209 or
        pooled["history_no_reserve"]["isolated_positive"] != 332):
        raise ValueError("Frozen Q4 denominators changed")
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed_offset":args.seed_offset,
            "question":"Does a head trained on actual isolated report sites transfer to excluded Bangkok districts?",
            "target":"Traffy report-positive 250m cell-days, not verified flooding",
            "isolated_definition":"No own or eight-adjacent report in the previous 365 days",
            "training_dates":["2023-01-01","2024-12-31"],
            "source_validation_dates":["2025-01-01","2025-09-30"],
            "geographic_test_dates":["2025-10-01","2025-12-31"],
            "queue":"10% of fold cells per day; 20% of slots reserved for isolated candidates",
            "training_note":"One representative per same-day touching positive component; unreported eligible comparisons are unknown, not verified dry",
            "features":"Pre-day 7/30/90/365-day report history; distant 365/90-day report smooths; SRTM terrain, OSM roads/canals, WorldCover cover. No future rain or coordinates.",
            "folds":records,"pooled_heldout_q4":pooled,
            "limitations":["Complaint buffers make even groups imperfect proxies for independent sites",
                           "Q4 2025 has been inspected repeatedly in prior experiments",
                           "Unreported cells and transparent satellite pixels are not verified dry",
                           "No rain-conditioned simulation or verified flood probability"]}
    out_json.write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"states":states},out_model)
    print("pooled",{k:(v["all_capture"],v["isolated_cell_capture"],
                        v["isolated_group_capture"]) for k,v in pooled.items()},flush=True)
    print("saved",out_json,out_model,flush=True)


if __name__=="__main__":
    main()
