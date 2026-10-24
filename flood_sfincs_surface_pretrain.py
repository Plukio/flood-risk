"""Train a 10 m CNN on paired uniform and mapped-roughness SFINCS cases.

The static Manning map comes from the audited WorldCover and OSM road probe.
Its numeric values are scenario assumptions, not measured coefficients. Pipe,
pump, infiltration and building-barrier effects are intentionally absent.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset
from torch import nn

from flood_sfincs_physics_model import PlainSpatialCNN
from flood_sfincs_physics_pretrain import (OUT as OLD, SIZE, STEP_S, _features,
                                         cases)
from flood_sfincs_surface_probe import OUT as PROBE


OUT = Path("artifacts/flood_bangkapi_surface/sfincs_surface_pretrain")
EPOCHS = 40
PATIENCE = 9


class SurfaceSpatialCNN(PlainSpatialCNN):
    """Same flow CNN and conservation rule; sixth channel is Manning n anomaly."""

    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(6, 48, 3, padding=1)


def features(z, previous, current, rate, manning):
    old = _features(z, previous, current, rate)
    return np.concatenate((old, ((manning/.035)-1)[None].astype(np.float32)))


def prepare():
    with np.load(PROBE / "surface_10m.npz") as d:
        n = d["assumed_manning_n"]
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    names = []
    for row, col, _, rate, minutes in cases():
        name = f"r{row}c{col}_{rate}mmh_{minutes}min"
        names.append(name)
        original = OLD / "cases" / name
        dest = OUT / "cases" / name
        dest.mkdir(parents=True, exist_ok=True)
        for fname in ("sfincs.dep", "sfincs.msk", "sfincs.prcp"):
            shutil.copyfile(original / fname, dest / fname)
        section = n[row*SIZE:(row+1)*SIZE, col*SIZE:(col+1)*SIZE]
        if section.shape != (SIZE, SIZE):
            raise ValueError(f"Surface grid does not cover {name}")
        np.flipud(section).T.astype("<f4").tofile(dest / "sfincs.man")
        inp = (original / "sfincs.inp").read_text()
        if "manning = 0.035\n" not in inp:
            raise ValueError(f"Expected original Manning setting in {name}")
        inp = inp.replace("manning = 0.035\n", "manningfile = sfincs.man\n")
        if "storevel" not in inp:
            inp += "storevel = 1\n"
        (dest / "sfincs.inp").write_text(inp)
    (OUT / "case_names.txt").write_text("\n".join(names)+"\n")
    (OUT / "run_all.sh").write_text("""#!/bin/sh
set -eu
count=0
while IFS= read -r name; do
  cd "/data/cases/$name"
  sfincs > sfincs.stdout 2>&1
  test -s sfincs_map.nc
  count=$((count + 1))
  if [ $((count % 20)) -eq 0 ]; then echo "Completed $count cases"; fi
done < /data/case_names.txt
echo "Completed $count cases total"
""")
    (OUT / "design.json").write_text(json.dumps({
        "cases": len(names), "teacher": "SFINCS v2.4.1 regular grid",
        "surface_source": str(PROBE / "surface_10m.npz"),
        "physical_change": "per-cell Manning n only",
        "paired_baseline": str(OLD / "sequences.npz"),
        "same_tile_storm_split_as_baseline": True,
        "closed_domain": True, "no_infiltration_or_drainage": True,
    }, indent=2)+"\n")
    print(f"Prepared {len(names)} paired mapped-roughness cases")


def collect():
    with np.load(PROBE / "surface_10m.npz") as d:
        n_full = d["assumed_manning_n"]
    depths, velocities, uniform_velocities, ns = [], [], [], []
    max_n_error = max_balance_mm = 0.
    for row, col, _, rate, minutes in cases():
        name = f"r{row}c{col}_{rate}mmh_{minutes}min"
        section = n_full[row*SIZE:(row+1)*SIZE, col*SIZE:(col+1)*SIZE]
        with Dataset(OUT / "cases" / name / "sfincs_map.nc") as ds:
            h = np.asarray(ds.variables["h"][:], np.float32)[:, ::-1, :]
            if "u" not in ds.variables or "v" not in ds.variables:
                raise ValueError(f"Missing saved velocity for {name}")
            u = np.asarray(ds.variables["u"][:], np.float32)[:, ::-1, :]
            v = np.asarray(ds.variables["v"][:], np.float32)[:, ::-1, :]
            saved_n = np.asarray(ds.variables["manning"][:], np.float32)[::-1]
        with Dataset(OLD / "cases" / name / "sfincs_map.nc") as ds:
            if "u" not in ds.variables or "v" not in ds.variables:
                raise ValueError(f"Missing uniform-run velocity for {name}")
            uniform_u = np.asarray(ds.variables["u"][:], np.float32)[:, ::-1, :]
            uniform_v = np.asarray(ds.variables["v"][:], np.float32)[:, ::-1, :]
        if h.shape != (minutes//5+1, SIZE, SIZE):
            raise ValueError(f"Unexpected depth output for {name}: {h.shape}")
        max_n_error = max(max_n_error, float(np.max(np.abs(saved_n-section))))
        for step in range(len(h)):
            expected = rate/1000*STEP_S/3600*step
            max_balance_mm = max(max_balance_mm,
                abs(float(h[step].mean()-expected))*1000)
        padded = np.zeros((5, SIZE, SIZE), np.float32)
        padded[:len(h)] = np.maximum(h, 0)
        padded_uv = np.zeros((5, 2, SIZE, SIZE), np.float32)
        padded_uv[:len(u), 0] = u
        padded_uv[:len(v), 1] = v
        padded_uniform_uv = np.zeros((5, 2, SIZE, SIZE), np.float32)
        padded_uniform_uv[:len(uniform_u), 0] = uniform_u
        padded_uniform_uv[:len(uniform_v), 1] = uniform_v
        depths.append(padded)
        velocities.append(padded_uv)
        uniform_velocities.append(padded_uniform_uv)
        ns.append(section)
    if max_n_error > 1e-6 or max_balance_mm > .02:
        raise ValueError(f"Teacher check failed: n={max_n_error}, balance={max_balance_mm}")
    np.savez_compressed(OUT / "mapped_sequences.npz", h=np.stack(depths),
                        uv=np.stack(velocities),
                        uniform_uv=np.stack(uniform_velocities),
                        manning=np.stack(ns))
    audit = {"cases": len(depths), "max_manning_alignment_error": max_n_error,
             "max_volume_balance_error_mean_depth_mm": max_balance_mm}
    (OUT / "collection_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


def load_data():
    with np.load(OLD / "sequences.npz") as d:
        old = {k: d[k] for k in d.files}
    with np.load(OUT / "mapped_sequences.npz") as d:
        mapped = {k: d[k] for k in d.files}
    if len(old["rate"]) != len(mapped["h"]):
        raise ValueError("Mapped and original cases differ")
    return old, mapped


def initialise(model):
    state = torch.load(OLD / "pretrained.pt", map_location="cpu",
                       weights_only=True)["states"]["plain_spatial_cnn"]
    first = state.pop("entry.weight")
    model.load_state_dict({**state, "entry.weight": model.entry.weight.detach()})
    with torch.no_grad():
        model.entry.weight[:, :5].copy_(first)
        model.entry.weight[:, 5:].zero_()


def train(extra_rows=None, output_dir=OUT):
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    old, mapped = load_data()
    n_uniform = np.full((SIZE, SIZE), .035, np.float32)
    rows = []
    for i in range(len(old["rate"])):
        for arm, depths, n in (("uniform", old["h"][i], n_uniform),
                               ("mapped", mapped["h"][i], mapped["manning"][i])):
            for step in range(1, int(old["length"][i])):
                x = features(old["z"][i], depths[max(0, step-2)],
                             depths[step-1], float(old["rate"][i]), n)
                rows.append((x, depths[step][None].copy(),
                             str(old["split"][i]), int(old["rate"][i]), arm))
    train_rows = [r for r in rows if r[2] == "train"]
    if extra_rows:
        train_rows.extend(extra_rows)
    valid_rows = [r for r in rows if r[2] == "validation" and r[3] != 60]
    model = SurfaceSpatialCNN().to(device)
    initialise(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.002)
    rng = np.random.default_rng(327)
    curve = []
    best = float("inf")
    best_epoch = 0
    best_state = None

    def batch_loss(batch, augmentation):
        x = torch.from_numpy(np.stack([r[0] for r in batch])).to(device)
        y = torch.from_numpy(np.stack([r[1] for r in batch])).to(device)
        if augmentation:
            k = int(rng.integers(4))
            x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
            if rng.random() < .5:
                x, y = torch.flip(x, (-1,)), torch.flip(y, (-1,))
        pred = model(x)
        absolute = torch.abs(pred-y)*1000
        weight = 1+2*(y/.02).clamp(0, 1)
        return (absolute*weight).mean(), float(absolute.mean().detach().cpu())

    print(f"device={device}; train={len(train_rows)}, val={len(valid_rows)}", flush=True)
    for epoch in range(1, EPOCHS+1):
        model.train()
        order = rng.permutation(len(train_rows))
        loss_values = []
        for start in range(0, len(order), 12):
            loss, _ = batch_loss([train_rows[j] for j in order[start:start+12]], True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_values.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            valid = np.mean([batch_loss(valid_rows[j:j+12], False)[1]
                             for j in range(0, len(valid_rows), 12)])
        curve.append({"epoch": epoch, "train_weighted_l1_mm": float(np.mean(loss_values)),
                      "validation_one_step_mae_mm": float(valid)})
        if valid < best:
            best, best_epoch = valid, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch} train={np.mean(loss_values):.3f} val={valid:.3f}", flush=True)
        if epoch-best_epoch >= PATIENCE:
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "architecture": "SurfaceSpatialCNN",
                "sixth_channel": "Manning n / 0.035 - 1"}, output_dir / "pretrained.pt")
    result = {"initialisation": "old plain CNN weights; sixth input weights zero",
              "best_epoch": best_epoch, "best_validation_one_step_mae_mm": best,
              "train_transitions": len(train_rows), "validation_transitions": len(valid_rows),
              "curve": curve}
    (output_dir / "training.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"Saved best epoch {best_epoch}: {best:.3f} mm")


def evaluate():
    torch.set_num_threads(4)
    old, mapped = load_data()
    old_model = PlainSpatialCNN().eval()
    old_state = torch.load(OLD / "pretrained.pt", map_location="cpu",
                           weights_only=True)["states"]["plain_spatial_cnn"]
    old_model.load_state_dict(old_state)
    model = SurfaceSpatialCNN().eval()
    model.load_state_dict(torch.load(OUT / "pretrained.pt", map_location="cpu",
                                     weights_only=True)["state"])
    n_uniform = np.full((SIZE, SIZE), .035, np.float32)
    buckets = {}
    for i in range(len(old["rate"])):
        split, rate = str(old["split"][i]), int(old["rate"][i])
        if split not in ("validation", "test"):
            continue
        group = f"{split}_{'unseen' if rate == 60 else 'seen'}"
        for arm, truth, n in (("uniform", old["h"][i], n_uniform),
                              ("mapped", mapped["h"][i], mapped["manning"][i])):
            key = f"{group}_{arm}"
            buckets.setdefault(key, {})
            for label, net, input_n in (("old_no_surface", old_model, None),
                                        ("surface_true_n", model, n),
                                        ("surface_uniform_n", model, n_uniform)):
                previous = np.zeros((SIZE, SIZE), np.float32)
                current = previous.copy()
                ones = []
                for step in range(1, int(old["length"][i])):
                    input_x = _features(old["z"][i], previous, current, rate) if input_n is None else (
                        features(old["z"][i], previous, current, rate, input_n))
                    with torch.no_grad():
                        following = net(torch.from_numpy(input_x[None]))[0, 0].numpy()
                    true_x = _features(old["z"][i], truth[max(0, step-2)],
                                       truth[step-1], rate) if input_n is None else (
                        features(old["z"][i], truth[max(0, step-2)],
                                 truth[step-1], rate, input_n))
                    with torch.no_grad():
                        one = net(torch.from_numpy(true_x[None]))[0, 0].numpy()
                    ones.append(float(np.mean(np.abs(one-truth[step]))*1000))
                    previous, current = current, following
                target = truth[int(old["length"][i])-1]
                wet = target > .02
                result = {"final_rollout_mae_mm": float(np.mean(np.abs(current-target))*1000),
                          "final_wet_gt20mm_mae_mm": float(np.mean(np.abs(current[wet]-target[wet]))*1000) if wet.any() else None,
                          "mean_one_step_mae_mm": float(np.mean(ones))}
                buckets[key].setdefault(label, []).append(result)
    summary = {}
    for key, arms in buckets.items():
        summary[key] = {}
        for label, rows in arms.items():
            summary[key][label] = {"cases": len(rows)}
            for metric in rows[0]:
                values = [r[metric] for r in rows if r[metric] is not None]
                summary[key][label][metric] = float(np.mean(values)) if values else None
    result = {"target": "synthetic SFINCS depth, not observed flooding",
              "evaluation": "geographically held-out tiles and unseen 60 mm/h storm",
              "results": summary}
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "collect", "train", "evaluate"))
    action = parser.parse_args().action
    {"prepare": prepare, "collect": collect, "train": train,
     "evaluate": evaluate}[action]()
