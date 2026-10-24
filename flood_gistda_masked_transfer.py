"""Test whether masked-water pretraining improves rainfall-only Bangkok transfer.

The fixed Bangkok target is fully hidden. Four other province patches are
reserved for geographic model selection, including normalization and epochs.
The auxiliary completion test reveals mapped water only outside its mask.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_gistda_masked_completion import (
    BLOCK, CompletionUNet, block_mask, capture, model_input, predict,
)
from flood_gistda_multiregion_data import OUTPUT, SIZE
from flood_gistda_multiregion_train import TRAIN_DATES, density_loss, normalize, scenes
from flood_gistda_province_2025_extra_train import append_2025
from flood_gistda_province_expand_bangkok_eval import bma_arrays
from flood_gistda_province_expand_train import TARGET, load_all


SEED = 105
GEO_VAL = ("pathum_thani", "chachoengsao", "ayutthaya_holdout", "nakhon_pathom")
RESIDUAL_SCALE = .4
RESULT = OUTPUT / "gistda_masked_transfer_bangkok_metrics.json"
PRETRAIN_CHECKPOINT = OUTPUT / "gistda_masked_transfer_pretrain.pt"
FINAL_CHECKPOINT = OUTPUT / "gistda_masked_transfer_final.pt"


def tensors(item: dict, crop: np.ndarray, di: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(item["x"][di:di + 1]), torch.from_numpy(crop[None])


def main_logits(model: CompletionUNet, x: torch.Tensor,
                crop: torch.Tensor) -> torch.Tensor:
    zero = x.new_zeros((len(x), 1, SIZE, SIZE))
    _, residual = model(x, zero, zero, crop)
    return torch.log(crop + .03) + RESIDUAL_SCALE * torch.tanh(residual)


def random_positive_mask(wet: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positive = np.argwhere(wet > 1)
    if len(positive) == 0:
        raise ValueError("Patch-day has no mapped positives")
    cell = positive[int(rng.integers(len(positive)))]
    row = int(np.clip(cell[0] - rng.integers(BLOCK), 0, SIZE - BLOCK))
    col = int(np.clip(cell[1] - rng.integers(BLOCK), 0, SIZE - BLOCK))
    return block_mask(row, col)


def fixed_positive_mask(wet: np.ndarray, name: str, di: int) -> np.ndarray:
    blocks = [(r, c) for r in range(0, SIZE, BLOCK)
              for c in range(0, SIZE, BLOCK)
              if wet[r:r + BLOCK, c:c + BLOCK].sum() > 1]
    row, col = blocks[(di + sum(map(ord, name))) % len(blocks)]
    return block_mask(row, col)


def masked_loss(model: CompletionUNet, item: dict, crop: np.ndarray,
                di: int, missing: np.ndarray) -> torch.Tensor:
    wet = item["wet"][di]
    x, prior = tensors(item, crop, di)
    observed, known = model_input(wet, missing, torch.device("cpu"))
    logits, residual = model(x, observed, known, prior)
    mask = torch.from_numpy(missing)
    truth = torch.from_numpy(wet[missing])
    return density_loss(logits[0][mask], truth) + .0005 * residual.square().mean()


@torch.no_grad()
def geo_mask_loss(model: CompletionUNet, regions: dict, crop: dict,
                  pairs: list) -> float:
    model.eval()
    values = [float(masked_loss(model, regions[n], crop[n], di,
                                fixed_positive_mask(regions[n]["wet"][di], n, di)))
              for n, di in pairs]
    return float(np.mean(values))


@torch.no_grad()
def geo_full_capture(model: CompletionUNet, regions: dict, crop: dict,
                     pairs: list) -> float:
    model.eval()
    total = 0.0
    caught = 0.0
    all_cells = np.ones((SIZE, SIZE), bool)
    for n, di in pairs:
        x, prior = tensors(regions[n], crop[n], di)
        score = main_logits(model, x, prior)[0].numpy()
        wet = regions[n]["wet"][di]
        total += float(wet.sum())
        caught += capture(score, wet, all_cells) * float(wet.sum())
    return caught / total


def pretrain(regions: dict, crop: dict, train_pairs: list,
             val_pairs: list) -> tuple[CompletionUNet, dict]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = CompletionUNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    curve = []
    for epoch in range(1, 51):
        model.train()
        losses = []
        for idx in rng.permutation(len(train_pairs)):
            name, di = train_pairs[int(idx)]
            mask = random_positive_mask(regions[name]["wet"][di], rng)
            loss = masked_loss(model, regions[name], crop[name], di, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val = geo_mask_loss(model, regions, crop, val_pairs)
        curve.append({"epoch": epoch, "train_mask_loss": round(float(np.mean(losses)), 5),
                      "geo_val_mask_loss": round(val, 5)})
        if val < best - .0005:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == 1:
            print("pretrain", curve[-1], "selected", best_epoch, flush=True)
        if epoch >= 20 and stale >= 10:
            break
    model.load_state_dict(best_state)
    return model, {"selected_epoch": best_epoch, "curve": curve}


def finetune(start: CompletionUNet | None, regions: dict, crop: dict,
             train_pairs: list, val_pairs: list, auxiliary: bool
             ) -> tuple[CompletionUNet, dict]:
    torch.manual_seed(SEED + (1 if auxiliary else 2))
    rng = np.random.default_rng(SEED + (1 if auxiliary else 2))
    model = CompletionUNet()
    if start is not None:
        model.load_state_dict(start.state_dict())
        # The no-context head starts at the exact crop ranking in both runs.
        torch.nn.init.zeros_(model.out.weight)
        torch.nn.init.zeros_(model.out.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    crop_baseline = geo_full_capture(model, regions, crop, val_pairs)
    best = crop_baseline
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stale = 0
    curve = []
    for epoch in range(1, 61):
        model.train()
        losses = []
        for idx in rng.permutation(len(train_pairs)):
            name, di = train_pairs[int(idx)]
            x, prior = tensors(regions[name], crop[name], di)
            wet = torch.from_numpy(regions[name]["wet"][di])
            score = main_logits(model, x, prior)[0]
            loss = density_loss(score, wet)
            if auxiliary and rng.random() < .25:
                missing = random_positive_mask(regions[name]["wet"][di], rng)
                loss = loss + .15 * masked_loss(model, regions[name], crop[name], di, missing)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val = geo_full_capture(model, regions, crop, val_pairs)
        curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
                      "geo_val_top_10pct_capture": round(val, 5)})
        if val > best + .0001:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == 1:
            print("finetune", "pretrained" if auxiliary else "scratch",
                  curve[-1], "selected", best_epoch, flush=True)
        if epoch >= 20 and stale >= 12:
            break
    model.load_state_dict(best_state)
    return model, {"selected_epoch": best_epoch,
                   "geo_val_crop_capture": round(crop_baseline, 5),
                   "geo_val_selected_capture": round(best, 5), "curve": curve}


@torch.no_grad()
def evaluate(model: CompletionUNet, target: dict, crop: np.ndarray,
             dates: list[str], area: np.ndarray, bma_wet: np.ndarray,
             masked: bool) -> dict:
    model.eval()
    bma = area > 1000
    full = []
    blocks = []
    for di, day in enumerate(dates[:8]):
        x, prior = tensors(target, crop, di)
        score = main_logits(model, x, prior)[0].numpy()
        wet = bma_wet[di]
        full.append({"date": str(day), "mapped_area_km2": float(wet[bma].sum()/1e6),
                     "model_capture": capture(score, wet, bma),
                     "crop_capture": capture(crop, wet, bma)})
        if not masked:
            continue
        for row in range(0, SIZE, BLOCK):
            for col in range(0, SIZE, BLOCK):
                missing = block_mask(row, col)
                candidate = bma & missing
                if candidate.sum() < 10 or wet[candidate].sum() <= 1:
                    continue
                completed = predict(model, target["x"][di], crop,
                                    target["wet"][di], missing, torch.device("cpu"))
                blocks.append({"date": str(day), "row": row, "col": col,
                               "mapped_area_km2": float(wet[candidate].sum()/1e6),
                               "model_capture": capture(completed, wet, candidate),
                               "crop_capture": capture(crop, wet, candidate)})
    result = {"full_dates": full}
    for name, rows in (("full", full), ("masked", blocks)):
        if not rows:
            continue
        weight = sum(r["mapped_area_km2"] for r in rows)
        result[name + "_summary"] = {"evaluations": len(rows),
            "model_capture": round(sum(r["model_capture"]*r["mapped_area_km2"]
                                       for r in rows)/weight, 5),
            "crop_capture": round(sum(r["crop_capture"]*r["mapped_area_km2"]
                                      for r in rows)/weight, 5)}
    return result


def main() -> None:
    torch.set_num_threads(4)
    regions, dates = load_all()
    crop = {name: item["x"][0, 17].copy() for name, item in regions.items()}
    dates = append_2025(regions, dates)
    train_regions = tuple(n for n in regions if n != TARGET and n not in GEO_VAL)
    normalizer = normalize(regions, train_regions)
    train_dates = TRAIN_DATES + tuple(range(8, len(dates)))
    train_pairs = scenes(regions, train_regions, train_dates)
    val_pairs = scenes(regions, GEO_VAL, tuple(range(8)))
    if not train_pairs or not val_pairs:
        raise ValueError("Missing positive training or geographic validation scenes")
    pre, pre_log = pretrain(regions, crop, train_pairs, val_pairs)
    torch.save({"state_dict": pre.state_dict(), "train_regions": train_regions,
                "geo_val_regions": GEO_VAL, "selected_epoch": pre_log["selected_epoch"]},
               PRETRAIN_CHECKPOINT)
    transfer, transfer_log = finetune(pre, regions, crop, train_pairs, val_pairs, True)
    scratch, scratch_log = finetune(None, regions, crop, train_pairs, val_pairs, False)
    area, bma_wet = bma_arrays(dates[:8])
    scores = {"pretrained_masked_only": evaluate(pre, regions[TARGET], crop[TARGET],
                                                  dates, area, bma_wet, True),
              "pretrained_then_multitask": evaluate(transfer, regions[TARGET],
                                                     crop[TARGET], dates, area, bma_wet, True),
              "scratch_full_only": evaluate(scratch, regions[TARGET], crop[TARGET],
                                             dates, area, bma_wet, False)}
    report = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "target": "GISTDA mapped positive area, no verified dry labels",
              "main_test": "held-out Nong Chok / BMA, full same-day map hidden",
              "auxiliary_test": "held-out Nong Chok / BMA, 20x20 cells hidden with same-day map outside block supplied",
              "train_regions": train_regions, "geo_val_regions": GEO_VAL,
              "train_patch_days": len(train_pairs), "geo_val_patch_days": len(val_pairs),
              "train_dates": [dates[i] for i in train_dates],
              "normalization": normalizer,
              "pretrain": pre_log, "transfer_finetune": transfer_log,
              "scratch_finetune": scratch_log, "bangkok": scores,
              "rain_caveat": "Full local-day rain can postdate GISTDA image; no rain counterfactual is validated."}
    RESULT.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"state_dict": transfer.state_dict(), "train_regions": train_regions,
                "geo_val_regions": GEO_VAL, "normalization": normalizer,
                "selected_epoch": transfer_log["selected_epoch"]}, FINAL_CHECKPOINT)
    print("SUMMARY", {k: v.get("full_summary") for k, v in scores.items()}, flush=True)
    print("MASKED", {k: v.get("masked_summary") for k, v in scores.items()}, flush=True)


if __name__ == "__main__":
    main()
