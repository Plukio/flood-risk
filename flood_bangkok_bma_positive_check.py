"""Check saved complaint models against conservatively located BMA wet sites.

BMA records are independent positive road-flood observations with reported
depth, but their event locations are matched to later inventory points.
There are no BMA dry-cell labels, so this reports positive-site rank only.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import torch
from scipy.stats import rankdata

from flood_bangkok_cnn_gnn_hybrid import FlowHybrid, load_graphs
from flood_bangkok_multitask_depth_cnn import matched_depths
from flood_bangkok_spatial_cnn_citywide import (
    ARTIFACT as BASE_ARTIFACT, CAL_N, CHANNELS, FEATURE_CACHE, HOLDOUT,
    cache_arrays, prepare_cache, transform,
)
from flood_bangkok_spatial_cnn_infrastructure import augmented_arrays
from flood_bangkok_transfer import load_districts
from flood_bangkapi_spatial_cnn import SpatialCNN
from flood_bangkapi_surface_model import OUTPUT, write_csv


ARTIFACT = "surface_bma_positive_site_check_citywide_v1"
START = date(2022, 1, 1)


def percentile(scores: np.ndarray, site_index: int) -> float:
    if len(scores) < 2:
        return 1.0
    # Higher scores and larger returned percentiles are better. Average ties.
    rank = rankdata(-scores, method="average")[site_index]
    return float(1 - (rank - 1) / (len(scores) - 1))


@torch.no_grad()
def scores(model: SpatialCNN, raw: np.ndarray, stats: list,
           rr: np.ndarray, cc: np.ndarray, device: torch.device,
           masked: tuple[int, ...] = ()) -> np.ndarray:
    x = torch.from_numpy(transform(raw, stats, masked)).to(device)
    model.eval()
    return model(x).cpu().numpy()[0, 0, rr, cc]


def main() -> None:
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    districts = {d["code"]: d for d in load_districts()}
    matched, records, audit = matched_depths(districts)
    meta = prepare_cache(districts)
    base_arrays = cache_arrays(tuple(districts))
    aug_arrays = augmented_arrays(meta)
    graphs = load_graphs(meta, device)
    base_ck = torch.load(OUTPUT / f"{BASE_ARTIFACT}_forty_two_more_steps.pt",
                         map_location=device, weights_only=False)
    canal_ck = torch.load(OUTPUT / "surface_crossdistrict_infrastructure_ablation_citywide_v1_canal_only.pt",
                          map_location=device, weights_only=False)
    graph_ck = torch.load(OUTPUT / "surface_crossdistrict_cnn_gnn_flow_v1.pt",
                          map_location=device, weights_only=False)
    base = SpatialCNN(CHANNELS + 1).to(device)
    base.load_state_dict(base_ck["state_dict"])
    canal = SpatialCNN(CHANNELS + 8).to(device)
    canal.load_state_dict(canal_ck["state_dict"])
    hybrid = FlowHybrid(SpatialCNN(CHANNELS + 1)).to(device)
    hybrid.load_state_dict(graph_ck["state_dict"])
    hybrid.eval()
    if base_ck["stats"] != graph_ck["stats"][:CHANNELS]:
        raise ValueError("Base and graph normalization differ")
    rows = []
    for (code, day, row, col), depth in sorted(matched.items()):
        if not (date(2025, 10, 1) <= day <= date(2025, 12, 31)):
            continue
        day_idx = (day - START).days
        rr = np.asarray(meta[code]["valid_rows"])
        cc = np.asarray(meta[code]["valid_cols"])
        site = np.flatnonzero((rr == row) & (cc == col))
        if len(site) != 1:
            continue
        site_idx = int(site[0])
        base_x, base_y = base_arrays[code]
        aug_x, _ = aug_arrays[code]
        x = torch.from_numpy(transform(aug_x[day_idx:day_idx + 1],
                                       graph_ck["stats"])).to(device)
        with torch.no_grad():
            graph_scores = hybrid(x, graphs[code])[0].cpu().numpy()[0, 0, rr, cc]
        original_scores = scores(base, base_x[day_idx:day_idx + 1],
                                 base_ck["stats"], rr, cc, device)
        canal_scores = scores(canal, aug_x[day_idx:day_idx + 1],
                              canal_ck["stats"], rr, cc, device,
                              tuple(canal_ck["masked_channels"]))
        history = np.asarray(base_y[1096:CAL_N, 0, rr, cc]).sum(axis=0)
        rows.append({"district_code": code, "date": str(day), "grid_row": row,
                     "grid_col": col, "depth_cm": depth,
                     "geographic_holdout": code in HOLDOUT,
                     "historical_percentile": round(percentile(history, site_idx), 5),
                     "base_cnn_percentile": round(percentile(original_scores, site_idx), 5),
                     "canal_cnn_percentile": round(percentile(canal_scores, site_idx), 5),
                     "cnn_gnn_percentile": round(percentile(graph_scores, site_idx), 5)})
    if not rows:
        raise ValueError("No exact-matched BMA test site-days on the model grid")
    def summary(group: list[dict]):
        return {"site_days": len(group),
                "districts": len({r["district_code"] for r in group}),
                "mean_rank_percentile": {
                    key: round(float(np.mean([r[key] for r in group])), 5)
                    for key in ("historical_percentile", "base_cnn_percentile",
                                "canal_cnn_percentile", "cnn_gnn_percentile")}}
    metrics = {"target": "BMA recorded positive major-road flood site-days in Q4 2025",
               "source": "https://data.bangkok.go.th/dataset/frd_dds",
               "location_method": "unique exact normalized road-area match to 2026 BMA flood-risk inventory point, same district polygon",
               "source_audit": audit,
               "all_matched": summary(rows),
               "geographic_holdouts": summary([r for r in rows if r["geographic_holdout"]]),
               "limitations": ["No BMA dry-cell observations; rank percentile is positive-only, not Spearman, AP or calibration",
                               "BMA archive covers selected major roads",
                               "Event locations are approximate 2026 inventory anchors",
                               "The 2025 test period has been reused across experiments"]}
    write_csv(OUTPUT / f"{ARTIFACT}_site_days.csv", rows)
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_matched": metrics["all_matched"],
                      "geographic_holdouts": metrics["geographic_holdouts"]},
                     indent=2))


if __name__ == "__main__":
    main()
