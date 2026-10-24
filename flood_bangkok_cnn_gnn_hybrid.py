"""Backtest a frozen citywide CNN with a learned drainage-graph residual.

The graph has terrain-inferred surface edges and OSM canal-way edges. It is
not a verified hydraulic network. The target remains Traffy complaint cells.

Run .venv/bin/python flood_bangkok_flow_graph.py first, then this script.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from flood_bangkok_flow_graph import GRAPH_DIR
from flood_bangkok_spatial_cnn_citywide import (
    ARTIFACT as BASE_ARTIFACT, BATCH, CAL_N, CHANNELS, HOLDOUT, POSITIVE_WEIGHT,
    SEED, TRAIN_N, prepare_cache, transform,
)
from flood_bangkok_spatial_cnn_infrastructure import augmented_arrays
from flood_bangkok_transfer import load_districts
from flood_bangkapi_proxy_probability import apply_calibration, calibrate_intercept, evaluation
from flood_bangkapi_spatial_cnn import SpatialCNN
from flood_bangkapi_surface_model import OUTPUT, rank_metrics, write_csv


ARTIFACT = "surface_crossdistrict_cnn_gnn_flow_v1"
INFRA_ARTIFACT = "surface_crossdistrict_infrastructure_cnn_citywide_v1"
STEPS_PER_EPOCH = 300
MAX_EPOCHS = 20
PATIENCE = 7


@dataclass
class Edges:
    source: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor


@dataclass
class FlowGraph:
    flat: torch.Tensor
    surface_down: Edges
    surface_up: Edges
    canal: Edges
    count: int
    shape: tuple[int, int]


def relation(pairs: np.ndarray, count: int, device: torch.device) -> Edges:
    source = torch.as_tensor(pairs[:, 0].astype(np.int64), device=device)
    target = torch.as_tensor(pairs[:, 1].astype(np.int64), device=device)
    degree = np.bincount(pairs[:, 1], minlength=count)
    weights = 1.0 / degree[pairs[:, 1]] if len(pairs) else np.empty(0)
    weight = torch.as_tensor(weights.astype(np.float32), device=device)
    return Edges(source, target, weight)


def load_graphs(meta: dict, device: torch.device) -> dict[str, FlowGraph]:
    result = {}
    for code, item in meta.items():
        with np.load(GRAPH_DIR / f"{code}.npz") as saved:
            rr, cc = saved["rows"], saved["cols"]
            surface, canal = saved["surface"], saved["canal"]
            h, w = map(int, saved["shape"])
        if (not np.array_equal(rr, item["valid_rows"]) or
                not np.array_equal(cc, item["valid_cols"]) or
                [h, w] != item["shape"][2:]):
            raise ValueError(f"Graph/grid mismatch in {code}")
        count = len(rr)
        flat = torch.as_tensor((rr * w + cc).astype(np.int64), device=device)
        result[code] = FlowGraph(
            flat, relation(surface, count, device),
            relation(surface[:, ::-1], count, device),
            relation(canal, count, device), count, (h, w))
    return result


def aggregate(h: torch.Tensor, edges: Edges) -> torch.Tensor:
    result = torch.zeros_like(h)
    if edges.source.numel():
        messages = h.index_select(1, edges.source) * edges.weight[None, :, None]
        result.index_add_(1, edges.target, messages)
    return result


class FlowHybrid(nn.Module):
    def __init__(self, baseline: SpatialCNN):
        super().__init__()
        self.baseline = baseline
        self.baseline.requires_grad_(False)
        width = 16
        self.project = nn.Linear(24 + 7, width)
        self.self_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(2))
        self.down_layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.up_layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.canal_layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, graph: FlowGraph) -> tuple[torch.Tensor, torch.Tensor]:
        base_x = torch.cat((x[:, :CHANNELS], x[:, -1:]), dim=1)
        with torch.no_grad():
            hidden = self.baseline.context[:4](base_x)
            baseline_logits = self.baseline(base_x)
        node_hidden = hidden.flatten(2).index_select(2, graph.flat).transpose(1, 2)
        assets = x[:, CHANNELS:CHANNELS + 7].flatten(2).index_select(
            2, graph.flat).transpose(1, 2)
        h = F.relu(self.project(torch.cat((node_hidden, assets), dim=-1)))
        for own, down, up, canal in zip(
                self.self_layers, self.down_layers, self.up_layers, self.canal_layers):
            h = F.relu(own(h) + down(aggregate(h, graph.surface_down)) +
                       up(aggregate(h, graph.surface_up)) +
                       canal(aggregate(h, graph.canal)))
        residual = self.output(h).squeeze(-1)
        dense = torch.zeros((x.shape[0], graph.shape[0] * graph.shape[1]),
                            device=x.device, dtype=x.dtype)
        dense.index_add_(1, graph.flat, residual)
        return baseline_logits + dense.reshape(x.shape[0], 1, *graph.shape), residual


@torch.no_grad()
def predict(model: FlowHybrid, raw: np.ndarray, stats: list,
            graph: FlowGraph, rr: np.ndarray, cc: np.ndarray,
            device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(raw), BATCH):
        x = torch.from_numpy(transform(raw[start:start + BATCH], stats)).to(device)
        logits, _ = model(x, graph)
        out.append(expit(logits.cpu().numpy()[:, 0, rr, cc]))
    return np.concatenate(out)


def train(arrays: dict, meta: dict, graphs: dict, codes: tuple[str, ...],
          stats: list, baseline: SpatialCNN, device: torch.device, artifact: str):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = FlowHybrid(baseline).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=0.001, weight_decay=0.001)
    bk_meta = meta["1006"]
    rr, cc = np.asarray(bk_meta["valid_rows"]), np.asarray(bk_meta["valid_cols"])
    bk_x, bk_y = arrays["1006"]
    truth = np.asarray(bk_y[TRAIN_N:CAL_N, 0, rr, cc]).ravel()
    best_ap, best_epoch, best_state, stale = -1.0, 0, None, 0
    curve = []
    pos_weight = torch.tensor(POSITIVE_WEIGHT, device=device)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for _ in range(STEPS_PER_EPOCH):
            code = codes[int(rng.integers(len(codes)))]
            raw_x, raw_y = arrays[code]
            ids = rng.integers(0, TRAIN_N, size=BATCH)
            x = torch.from_numpy(transform(raw_x[ids], stats)).to(device)
            y = torch.from_numpy(np.asarray(raw_y[ids], dtype=np.float32)).to(device)
            mask = x[:, -1:]
            optimizer.zero_grad()
            logits, residual = model(x, graphs[code])
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight, reduction="none")
            loss = (loss * mask).sum() / mask.sum() + 0.001 * residual.square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = predict(model, bk_x[TRAIN_N:CAL_N], stats,
                        graphs["1006"], rr, cc, device).ravel()
        ap = float(average_precision_score(truth, score))
        curve.append({"epoch": epoch, "train_loss_mean": round(float(np.mean(losses)), 6),
                      "bang_kapi_cal_ap": round(ap, 6)})
        if ap > best_ap + 1e-6:
            best_ap, best_epoch = ap, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"epoch {epoch}: loss={curve[-1]['train_loss_mean']:.5f} "
              f"Bang Kapi cal AP={ap:.5f} best={best_ap:.5f}", flush=True)
        if stale >= PATIENCE and epoch >= 10:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "stats": stats, "codes": codes,
                "selected_epoch": best_epoch}, OUTPUT / f"{artifact}.pt")
    return model, {"training_codes": list(codes), "selected_epoch": best_epoch,
                   "selection_bang_kapi_cal_ap": round(best_ap, 6),
                   "steps_per_epoch": STEPS_PER_EPOCH, "batch": BATCH,
                   "max_epochs": MAX_EPOCHS, "early_stop_patience": PATIENCE,
                   "curve": curve, "device": str(device)}


def evaluate(model: FlowHybrid, arrays: dict, meta: dict, graphs: dict,
             stats: list, codes: tuple[str, ...], device: torch.device) -> list[dict]:
    rows = []
    for code in codes:
        x, y = arrays[code]
        rr, cc = np.asarray(meta[code]["valid_rows"]), np.asarray(meta[code]["valid_cols"])
        cal_y = np.asarray(y[TRAIN_N:CAL_N, 0, rr, cc], dtype=np.uint8)
        test_y = np.asarray(y[CAL_N:, 0, rr, cc], dtype=np.uint8)
        raw_cal = predict(model, x[TRAIN_N:CAL_N], stats, graphs[code], rr, cc, device)
        raw_test = predict(model, x[CAL_N:], stats, graphs[code], rr, cc, device)
        if cal_y.sum() and cal_y.sum() < cal_y.size:
            bias = calibrate_intercept(raw_cal.ravel(), cal_y.ravel())
            calibrated = apply_calibration(raw_test, bias)
            calibration = "local Aug-Sep 2025"
        else:
            bias, calibrated = 0.0, raw_test
            calibration = "none: calibration period has one class"
        historical = np.asarray(y[1096:CAL_N, 0, rr, cc]).sum(axis=0)
        observed = test_y.sum(axis=0)
        expected = calibrated.sum(axis=0)
        rank = rank_metrics(expected, observed) if observed.sum() else None
        history = rank_metrics(historical, observed) if observed.sum() else None
        daily = evaluation(calibrated.ravel(), test_y.ravel()) if observed.sum() else None
        rows.append({"code": code, "district": meta[code]["name"],
                     "cells": len(rr), "cal_positive_cell_days": int(cal_y.sum()),
                     "test_positive_cell_days": int(test_y.sum()),
                     "historical_spearman": history["spearman"] if history else None,
                     "cnn_spearman": rank["spearman"] if rank else None,
                     "test_daily_ap": daily["average_precision"] if daily else None,
                     "predicted_positive_cell_days": daily["predicted_positive_cell_days"] if daily else None,
                     "calibration": calibration, "log_odds_shift": round(bias, 4)})
        print(f"test {code}: hybrid={rows[-1]['cnn_spearman']} "
              f"positives={int(test_y.sum())}", flush=True)
    return rows


def main(no_edges: bool = False) -> None:
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    districts = {d["code"]: d for d in load_districts()}
    meta = prepare_cache(districts)
    arrays = augmented_arrays(meta)
    graphs = load_graphs(meta, device)
    artifact = ARTIFACT + ("_no_edges" if no_edges else "")
    if no_edges:
        empty = relation(np.empty((0, 2), dtype=np.int32), 0, device)
        for graph in graphs.values():
            graph.surface_down = empty
            graph.surface_up = empty
            graph.canal = empty
    if int(np.asarray(arrays["1006"][1][CAL_N:]).sum()) != 242:
        raise ValueError("Bang Kapi labels changed")
    base_path = OUTPUT / f"{BASE_ARTIFACT}_forty_two_more_steps.pt"
    base_checkpoint = torch.load(base_path, map_location=device, weights_only=False)
    infra_checkpoint = torch.load(
        OUTPUT / f"{INFRA_ARTIFACT}_forty_two_infrastructure_long.pt",
        map_location="cpu", weights_only=False)
    stats = infra_checkpoint["stats"]
    if len(stats) != CHANNELS + 7 or stats[:CHANNELS] != base_checkpoint["stats"]:
        raise ValueError("Base and infrastructure normalization differ")
    baseline = SpatialCNN(CHANNELS + 1).to(device)
    baseline.load_state_dict(base_checkpoint["state_dict"])
    codes = tuple(code for code in districts if code not in HOLDOUT)
    if tuple(base_checkpoint["codes"]) != codes:
        raise ValueError("Training districts differ from reference CNN")
    model, training = train(arrays, meta, graphs, codes, stats, baseline, device, artifact)
    rows = evaluate(model, arrays, meta, graphs, stats, tuple(districts), device)
    reference = json.loads((OUTPUT / f"{BASE_ARTIFACT}_metrics.json").read_text())[
        "models"]["forty_two_more_steps"]["evaluated_districts"]
    by_code = {row["code"]: row for row in reference}
    for row in rows:
        old = by_code[row["code"]]
        if (row["cells"] != old["cells"] or
                row["test_positive_cell_days"] != old["test_positive_cell_days"] or
                row["cal_positive_cell_days"] != old["cal_positive_cell_days"]):
            raise ValueError(f"Outcome mismatch for {row['code']}")
    write_csv(OUTPUT / f"{artifact}_district_comparison.csv", [
        {"group": "geographic_holdout" if r["code"] in HOLDOUT else "training_district",
         "baseline_spearman": by_code[r["code"]]["cnn_spearman"],
         "baseline_daily_ap": by_code[r["code"]]["test_daily_ap"], **r}
        for r in rows])
    payload = {
        "target": "100 m direct Traffy flood-complaint buffer touches a 250 m cell-day",
        "not_target": "verified inundation or flood depth",
        "graph_audit": str(GRAPH_DIR / "audit.json"),
        "baseline_checkpoint": str(base_path),
        "method": ("frozen 17-channel CNN plus a local node MLP residual with no graph edges"
                   if no_edges else
                   "frozen 17-channel CNN plus two relation-specific graph layers and a learned logit residual"),
        "no_edges_control": no_edges,
        "train_period": ["2022-01-01", "2025-07-31"],
        "selection_and_local_calibration": ["2025-08-01", "2025-09-30"],
        "test_period": ["2025-10-01", "2025-12-31"],
        "geographic_holdout_codes": list(HOLDOUT), "training": training,
        "evaluated_districts": rows,
        "limitations": ["Surface edges are terrain-inferred; canal edges are mapped geometry, not verified hydraulic links",
                        "District graph omits cross-boundary connections",
                        "Graph has no pipe topology or time-matched gate/pump state",
                        "Open-data outcome is a complaint proxy, not verified flood/no-flood",
                        "2025 test period has been reused in prior experiments"]}
    (OUTPUT / f"{artifact}_metrics.json").write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-edges", action="store_true",
                        help="Matched node-MLP control with all graph connections removed")
    main(parser.parse_args().no_edges)
