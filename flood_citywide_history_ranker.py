"""Train a Bangkok-wide pre-Q4 report ranker and compare with 365-day history.

Model selection uses January–September 2025 after training on 2023–2024.
The selected recipe is then refit through September before scoring Q4 2025.
Unknown/unreported cells are sampled as comparison cells, not verified dry.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from shapely.geometry import box
from shapely.ops import unary_union

from flood_bangkok_transfer import CACHE as BMA_CACHE, load_districts
from flood_bangkapi_surface_model import rainfall_by_date
from flood_multi_target_citywide_data import OUT, COLS, ROWS, specs_and_masks
from flood_multi_target_citywide_eval import district_assignment, paired_interval
from flood_multi_target_data import START, complaint_grid
from flood_open_weak_bangkok_holdout_eval import bma_fraction, selected
from flood_open_weak_data import all_specs, load_static


TRAIN_START = (date(2023, 1, 1) - START).days
VALID_START = (date(2025, 1, 1) - START).days
TEST_START = (date(2025, 10, 1) - START).days
N_DAYS = (date(2025, 12, 31) - START).days + 1
SHAPE = (ROWS*80, COLS*80)
OUT_JSON = OUT / "q4_2025_history_ranker.json"
FEATURES = ("log_history_365d", "log_history_90d", "log_history_30d",
            "log_history_7d", "log_nearby_365d_250m",
            "log_nearby_365d_750m", "log_nearby_90d_250m",
            "srtm_elevation_m", "srtm_relative_1250m_m", "srtm_slope",
            "osm_canal_length_log", "osm_canal_distance_log",
            "osm_road_length_log", "osm_road_distance_log",
            "worldcover_built_fraction_2021", "worldcover_crop_fraction_2021",
            "worldcover_tree_grass_fraction_2021", "worldcover_bare_fraction_2021",
            "worldcover_wetland_fraction_2021", "worldcover_water_fraction_2021",
            "month_sin", "month_cos",
            "rain_today_mm_log", "rain_peak_3h_mm_log",
            "rain_prev24h_mm_log", "rain_prev72h_mm_log")


def load_grid():
    specs, _ = specs_and_masks()
    districts = load_districts()
    bma = unary_union([d["polygon"] for d in districts])
    original = all_specs()
    trained = unary_union([box(*original[n]["utm_bounds"])
                           for n in ("bang_kapi_core", "nong_chok")])
    excluded = trained.buffer(750)
    reports = np.zeros((N_DAYS, *SHAPE), np.uint8)
    static = np.zeros((13, *SHAPE), np.float32)
    candidate = np.zeros(SHAPE, bool)
    new = np.zeros(SHAPE, bool)
    district_id = np.full(SHAPE, -1, np.int16)
    tile_id = np.full(SHAPE, -1, np.int8)
    rain_by_tile = np.empty((N_DAYS, len(specs), 4), np.float32)
    last_day = START + timedelta(days=N_DAYS-1)
    for tile_number,(name, spec) in enumerate(specs.items()):
        row, col = [int(s[1:]) for s in name.split("_")[1:]]
        rr = slice((ROWS-1-row)*80, (ROWS-row)*80)
        cc = slice(col*80, (col+1)*80)
        mask = bma_fraction(spec, bma) > .02
        labels, _ = complaint_grid(name, spec, districts, N_DAYS)
        labels[:, ~mask] = 0
        reports[:, rr, cc] = labels
        static[:, rr, cc] = load_static(name)
        candidate[rr, cc] = mask
        new[rr, cc] = mask & (bma_fraction(spec, excluded) == 0)
        district_id[rr, cc] = district_assignment(spec, districts, mask)
        tile_id[rr, cc] = tile_number
        point = bma.intersection(box(*spec["utm_bounds"])).centroid
        rain_district = min(districts, key=lambda d:
                            d["polygon"].representative_point().distance(point))
        observed = rainfall_by_date(START,last_day,
            BMA_CACHE / f"rain_{rain_district['code']}_2021_2025.json")
        for day in range(N_DAYS):
            values = observed[START + timedelta(days=day)]
            rain_by_tile[day,tile_number] = np.log1p([
                values[k] for k in ("rain_today_mm", "rain_peak_3h_mm",
                                    "rain_prev24h_mm", "rain_prev72h_mm")])
    if (int(reports[TEST_START:, candidate].sum()) != 7209 or
        candidate.sum() != 25812 or new.sum() != 13302 or
        np.any(candidate & (district_id < 0))):
        raise ValueError("Citywide labels or geometry differ from frozen comparison")
    prefix = np.concatenate((np.zeros((1, *SHAPE), np.uint16),
                             np.cumsum(reports, axis=0, dtype=np.uint16)), axis=0)
    for day in (TRAIN_START, VALID_START, TEST_START, N_DAYS-1):
        if not np.array_equal(prefix[day]-prefix[day-365],
                              reports[day-365:day].sum(axis=0)):
            raise ValueError(f"Report history includes the scored day: {day}")
    return {"reports": reports, "prefix": prefix,
            "static": static[:, candidate].T.copy(),
            "candidate": candidate, "new": new,
            "district_id": district_id, "districts": districts,
            "tile_id": tile_id[candidate], "rain_by_tile": rain_by_tile,
            "candidate_flat": np.flatnonzero(candidate.ravel())}


def day_features(data: dict, day: int):
    prefix = data["prefix"]
    counts = {window: (prefix[day] - prefix[day-window]).astype(np.float32)
              for window in (365, 90, 30, 7)}
    near365 = gaussian_filter(counts[365], 1, mode="constant")
    wide365 = gaussian_filter(counts[365], 3, mode="constant")
    near90 = gaussian_filter(counts[90], 1, mode="constant")
    mask = data["candidate"]
    month = (START.fromordinal(START.toordinal()+day)).month
    calendar = np.array([math.sin(2*math.pi*month/12),
                         math.cos(2*math.pi*month/12)], np.float32)
    dynamic = np.stack([np.log1p(counts[k][mask]) for k in (365,90,30,7)] +
                       [np.log1p(x[mask]) for x in (near365,wide365,near90)], axis=1)
    features = np.concatenate((dynamic, data["static"],
                               np.broadcast_to(calendar, (len(dynamic),2)),
                               data["rain_by_tile"][day,data["tile_id"]]), axis=1)
    if features.shape[1] != len(FEATURES) or not np.isfinite(features).all():
        raise ValueError("Invalid training features")
    return features.astype(np.float32), counts[365][mask], counts[90][mask], near365[mask]


def sample_training(data: dict, days: range, seed: int):
    rng = np.random.default_rng(seed)
    samples, targets = [], []
    candidate_flat = data["candidate_flat"]
    for day in days:
        x, h365, _, _ = day_features(data, day)
        y = data["reports"][day].ravel()[candidate_flat]
        positive = np.flatnonzero(y)
        unlabeled = np.flatnonzero(~y.astype(bool))
        random_negative = rng.choice(unlabeled, size=min(256,len(unlabeled)), replace=False)
        # Hard comparisons near the currently highest-ranked historic sites.
        hot = unlabeled[np.argpartition(h365[unlabeled], -min(2048,len(unlabeled)))[-min(2048,len(unlabeled)):]]
        hard_negative = rng.choice(hot, size=min(256,len(hot)), replace=False)
        chosen = np.unique(np.concatenate((positive, random_negative, hard_negative)))
        samples.append(x[chosen])
        targets.append(y[chosen])
        if day % 90 == 0:
            print("training samples through", str(START.fromordinal(START.toordinal()+day)), flush=True)
    x = np.concatenate(samples)
    y = np.concatenate(targets)
    if not y.any() or y.all():
        raise ValueError("No class contrast in sampled training data")
    return x, y


def fit(x: np.ndarray, y: np.ndarray):
    model = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, min_samples_leaf=60,
        learning_rate=.07, l2_regularization=3., random_state=97)
    model.fit(x, y)
    return model


def score_days(data: dict, model, days: range):
    names = ("history_365d", "history_plus_recent",
             "history_plus_nearby", "history_recent_nearby",
             "history_recent_nearby_rain_010",
             "history_recent_nearby_rain_025",
             "history_recent_nearby_rain_050",
             "hgb", "hgb_history_025", "hgb_history_050",
             "hgb_history_100", "hgb_history_200")
    out = {key: [] for key in names}
    labels = []
    for day in days:
        x, h365, h90, near365 = day_features(data, day)
        log365, log90, lognear = np.log1p(h365), np.log1p(h90), np.log1p(near365)
        probability = model.predict_proba(x)[:,1]
        logit = np.log(np.clip(probability, 1e-6, 1-1e-6) /
                       np.clip(1-probability, 1e-6, 1))
        base = log365 + .4*log90 + .4*lognear
        rain = data["rain_by_tile"][day,data["tile_id"],0]
        scores = {"history_365d": h365,
                  "history_plus_recent": log365 + .5*log90,
                  "history_plus_nearby": log365 + .5*lognear,
                  "history_recent_nearby": base,
                  "history_recent_nearby_rain_010": base + .1*rain,
                  "history_recent_nearby_rain_025": base + .25*rain,
                  "history_recent_nearby_rain_050": base + .5*rain,
                  "hgb": logit,
                  "hgb_history_025": logit + .25*log365,
                  "hgb_history_050": logit + .5*log365,
                  "hgb_history_100": logit + log365,
                  "hgb_history_200": logit + 2*log365}
        for key in names:
            out[key].append(scores[key].astype(np.float32))
        labels.append(data["reports"][day].ravel()[data["candidate_flat"]])
        if day % 90 == 0:
            print("scored through", str(START.fromordinal(START.toordinal()+day)), flush=True)
    return {key: np.stack(value) for key,value in out.items()}, np.stack(labels)


def capture(score: np.ndarray, truth: np.ndarray, mask: np.ndarray):
    score, truth = score[:,mask], truth[:,mask]
    positives = truth.sum(axis=1).astype(np.int64)
    hits = np.array([np.dot(selected(s,.1),y) for s,y in zip(score,truth)], np.float64)
    total = int(positives.sum())
    return {"candidate_cells": int(mask.sum()),
            "positive_cell_days": total,
            "top_10pct_capture": round(float(hits.sum()/total),5) if total else None,
            "daily_positive": positives.tolist(),
            "daily_captured": np.round(hits,5).tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_grid()
    print("loaded",int(data["candidate"].sum()),"candidate cells",flush=True)
    x_train,y_train=sample_training(data,range(TRAIN_START,VALID_START),97)
    print("initial training",x_train.shape,int(y_train.sum()),flush=True)
    initial=fit(x_train,y_train)
    validation_scores,validation_truth=score_days(data,initial,range(VALID_START,TEST_START))
    all_candidates=np.ones(validation_truth.shape[1],bool)
    validation={name:capture(score,validation_truth,all_candidates)
                for name,score in validation_scores.items()}
    baseline=validation["history_365d"]["top_10pct_capture"]
    # Select the strongest recipe using pre-Q4 data only. If no candidate
    # exceeds history, retain history as the selected model.
    chosen=max(validation,key=lambda k:validation[k]["top_10pct_capture"])
    print("validation baseline",baseline,"chosen",chosen,
          validation[chosen]["top_10pct_capture"],flush=True)
    if chosen.startswith("hgb"):
        x_more,y_more=sample_training(data,range(VALID_START,TEST_START),98)
        final=fit(np.concatenate((x_train,x_more)),np.concatenate((y_train,y_more)))
    else:
        final=initial
    test_scores,test_truth=score_days(data,final,range(TEST_START,N_DAYS))
    baseline_q4=capture(test_scores["history_365d"],test_truth,
                        np.ones(test_truth.shape[1],bool))
    if abs(baseline_q4["top_10pct_capture"]-.47183)>1e-5:
        raise ValueError("Q4 history baseline differs from frozen citywide evaluation")
    new=data["new"].ravel()[data["candidate_flat"]]
    if abs(capture(test_scores["history_365d"],test_truth,new)["top_10pct_capture"]-.44925)>1e-5:
        raise ValueError("Q4 new-geography baseline differs from frozen comparison")
    by_scope={}
    for scope,mask in (("citywide",np.ones(test_truth.shape[1],bool)),
                       ("outside_750m_training_buffer",new)):
        rows={name:capture(test_scores[name],test_truth,mask)
              for name in dict.fromkeys(("history_365d", "history_recent_nearby", chosen))}
        rows["chosen_minus_history_95pct_weekly_block_interval"] = paired_interval(
            rows[chosen],rows["history_365d"])
        rows["chosen_minus_no_rain_95pct_weekly_block_interval"] = paired_interval(
            rows[chosen],rows["history_recent_nearby"])
        by_scope[scope]=rows
    district_ids=data["district_id"].ravel()[data["candidate_flat"]]
    district_results={}
    for i,district in enumerate(data["districts"]):
        mask=district_ids==i
        district_results[district["code"]]={
            "name":district["name_en"],
            "chosen":capture(test_scores[chosen],test_truth,mask),
            "history_365d":capture(test_scores["history_365d"],test_truth,mask),
            "new_geography_chosen":capture(test_scores[chosen],test_truth,mask & new),
            "new_geography_history_365d":capture(test_scores["history_365d"],test_truth,mask & new)}
    result={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metric":"daily top-10% Traffy report-positive cell-day capture; same citywide or district queue for compared scores",
            "training_period":[str(START.fromordinal(START.toordinal()+TRAIN_START)),"2024-12-31"],
            "validation_period":["2025-01-01","2025-09-30"],
            "final_training_period":["2023-01-01","2025-09-30"] if chosen.startswith("hgb") else None,
            "test_period":["2025-10-01","2025-12-31"],
            "features":FEATURES,"selected_recipe":chosen,
            "selected_recipe_formula":"log1p(prior 365-day report count) + 0.4*log1p(prior 90-day report count) + 0.4*log1p(Gaussian-smoothed prior 365-day report count, sigma 250 m) + 0.25*log1p(full local calendar-day rain mm at nearest district proxy)",
            "training_sample_rows":int(len(x_train)),
            "validation_capture":{k:v["top_10pct_capture"] for k,v in validation.items()},
            "q4_scopes":by_scope,"q4_districts":district_results,
            "limitations":"Unreported is unknown, not verified dry. Q4 was inspected in prior model work. This model does not use rain or future observations, but static land-cover/OSM snapshots may differ from historic dates."}
    OUT_JSON.write_text(json.dumps(result,indent=2)+"\n")
    print("Q4",{k:{s:v["top_10pct_capture"] for s,v in row.items() if isinstance(v,dict)}
                 for k,row in by_scope.items()},flush=True)
    print("saved",OUT_JSON,flush=True)


if __name__=="__main__":
    main()
