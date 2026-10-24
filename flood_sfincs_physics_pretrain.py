"""Pretrain a conservative spatial CNN on SFINCS 10 m water-state sequences.

Only synthetic, closed-boundary SFINCS states are targets here. Entire 960 m
tiles are held out geographically; an unseen rain rate and longer storm test
forcing transfer. This does not use Bangkok flood reports or measured depths.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

from flood_sfincs_lidar_trial import DTM, IMAGE


OUT = Path("artifacts/flood_bangkapi_surface/sfincs_physics_pretrain")
SIZE = 96
STEP_S = 300
# Training rates and durations: a 60 mm/h, 20 min scenario is reserved.
TRAIN_STORMS = ((20, 15), (40, 15), (80, 15))
UNSEEN_STORM = (60, 20)
GRID_ROWS = GRID_COLS = 9
MAX_EPOCHS = 80
PATIENCE = 15


def split_for_column(col: int) -> str:
    return "train" if col < 6 else ("validation" if col == 6 else "test")


def cases():
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            split = split_for_column(col)
            for rate, minutes in TRAIN_STORMS:
                yield row, col, split, rate, minutes
            if split != "train":
                yield row, col, split, *UNSEEN_STORM


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(exist_ok=True)
    lines = []
    with rasterio.open(DTM) as src:
        if src.crs.to_epsg() != 32647 or min(src.shape) < SIZE*9:
            raise ValueError("Expected Bang Kapi 10 m DTM and 9 by 9 tiles")
        for row, col, split, rate, minutes in cases():
            name = f"r{row}c{col}_{rate}mmh_{minutes}min"
            dest = OUT / "cases" / name
            dest.mkdir(exist_ok=True)
            window = Window(col*SIZE, row*SIZE, SIZE, SIZE)
            z = src.read(1, window=window).astype(np.float64)
            if not np.isfinite(z).all():
                raise ValueError(f"Invalid DTM tile {row},{col}")
            t = src.window_transform(window)
            x0, y0 = t.c, t.f-10*SIZE
            duration_s = minutes*60
            end = datetime(2026, 10, 23)+timedelta(seconds=duration_s)
            np.savetxt(dest / "sfincs.dep", np.flipud(z), fmt="%.8f")
            np.savetxt(dest / "sfincs.msk", np.ones_like(z, np.int8), fmt="%d")
            (dest / "sfincs.prcp").write_text(f"0 {rate}\n{duration_s} {rate}\n")
            (dest / "sfincs.inp").write_text(f"""x0 = {x0:.3f}
y0 = {y0:.3f}
mmax = {SIZE}
nmax = {SIZE}
dx = 10
dy = 10
rotation = 0
epsg = 32647
tref = 20261023 000000
tstart = 20261023 000000
tstop = {end:%Y%m%d %H%M%S}
depfile = sfincs.dep
mskfile = sfincs.msk
precipfile = sfincs.prcp
inputformat = asc
outputformat = net
zsini = {float(z.min())-1:.8f}
advection = 0
alpha = 0.4
huthresh = 0.0001
theta = 1.0
viscosity = 0
manning = 0.035
qinf = 0
dtmax = 2.5
dtout = {STEP_S}
dtmaxout = {duration_s}
storecumprcp = 1
""")
            lines.append(name)
    (OUT / "case_names.txt").write_text("\n".join(lines)+"\n")
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
    metadata = {"image": IMAGE, "source_dtm": str(DTM), "grid_size": SIZE,
                "cell_m": 10, "time_step_s": STEP_S,
                "training_storms_mm_h_minutes": TRAIN_STORMS,
                "unseen_storm_mm_h_minutes": UNSEEN_STORM,
                "geographic_split": {"train_cols": list(range(6)),
                                     "validation_cols": [6], "test_cols": [7, 8]},
                "cases": len(lines), "boundary": "closed", "infiltration": 0,
                "manning": .035}
    (OUT / "design.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Prepared {len(lines)} SFINCS cases in {OUT}")


def collect() -> None:
    from netCDF4 import Dataset

    elevations, depths, rates, lengths, rows, cols, splits, names = [], [], [], [], [], [], [], []
    max_balance_mm = 0.
    max_terrain_m = 0.
    for row, col, split, rate, minutes in cases():
        name = f"r{row}c{col}_{rate}mmh_{minutes}min"
        path = OUT / "cases" / name
        z = np.flipud(np.loadtxt(path / "sfincs.dep", dtype=np.float32))
        with Dataset(path / "sfincs_map.nc") as ds:
            h = np.asarray(ds.variables["h"][:], np.float32)[:, ::-1, :]
            z_out = np.asarray(ds.variables["zb"][:], np.float32)[::-1]
            if h.shape != (minutes//5+1, SIZE, SIZE):
                raise ValueError(f"Unexpected output times in {name}: {h.shape}")
            max_terrain_m = max(max_terrain_m, float(np.max(np.abs(z_out-z))))
        if not np.isfinite(h).all() or h.min() < -1e-5:
            raise ValueError(f"Invalid water depths in {name}")
        h = np.maximum(h, 0)
        for step in range(h.shape[0]):
            expected = rate/1000*STEP_S/3600*step
            max_balance_mm = max(max_balance_mm,
                float(abs(h[step].mean()-expected)*1000))
        padded = np.zeros((5, SIZE, SIZE), np.float32)
        padded[:len(h)] = h
        elevations.append(z)
        depths.append(padded)
        rates.append(rate)
        lengths.append(len(h))
        rows.append(row)
        cols.append(col)
        splits.append(split)
        names.append(name)
    if max_balance_mm > .02 or max_terrain_m > 1e-5:
        raise ValueError(f"SFINCS water/terrain audit failed: {max_balance_mm}, {max_terrain_m}")
    np.savez_compressed(OUT / "sequences.npz",
        z=np.stack(elevations), h=np.stack(depths), rate=np.asarray(rates),
        length=np.asarray(lengths), row=np.asarray(rows), col=np.asarray(cols),
        split=np.asarray(splits), name=np.asarray(names))
    audit = {"cases": len(names), "transitions": int(sum(np.asarray(lengths)-1)),
             "max_volume_balance_error_mean_depth_mm": max_balance_mm,
             "max_terrain_mismatch_m": max_terrain_m,
             "splits": {s: int(sum(x == s for x in splits)) for s in
                        ("train", "validation", "test")}}
    (OUT / "collection_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


def _features(z: np.ndarray, h_prev: np.ndarray, h_now: np.ndarray,
              rate: float) -> np.ndarray:
    centered = np.clip((z-np.median(z))/.5, -6, 6)
    relief = np.clip((z-gaussian_filter(z, 3))/.1, -6, 6)
    return np.stack((centered, relief, h_prev/.05, h_now/.05,
                     np.full_like(z, rate/60)), axis=0).astype(np.float32)


def train() -> None:
    import torch
    from torch import nn
    from torch.nn import functional as F

    from flood_sfincs_physics_model import SpatialFlowUNet, PlainSpatialCNN, LocalFlowMLP

    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with np.load(OUT / "sequences.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    seq = []
    for i in range(len(data["rate"])):
        for step in range(1, int(data["length"][i])):
            rate = float(data["rate"][i])
            z = data["z"][i]
            prior = data["h"][i, max(0, step-2)]
            current = data["h"][i, step-1]
            target = data["h"][i, step]
            x = _features(z, prior, current, rate)
            seq.append((x, target[None].copy(), str(data["split"][i]),
                        rate, int(data["row"][i]), int(data["col"][i])))
    train_rows = [r for r in seq if r[2] == "train"]
    valid_rows = [r for r in seq if r[2] == "validation" and r[3] != 60]
    if not train_rows or not valid_rows:
        raise ValueError("Missing training or validation sequences")
    print(f"device={device}, train transitions={len(train_rows)}, validation={len(valid_rows)}", flush=True)

    def batch_loss(model, batch):
        x = torch.from_numpy(np.stack([r[0] for r in batch])).to(device)
        y = torch.from_numpy(np.stack([r[1] for r in batch])).to(device)
        if model.training:
            k = int(rng.integers(4))
            x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
            if rng.random() < .5:
                x, y = torch.flip(x, (-1,)), torch.flip(y, (-1,))
        pred = model(x)
        diff = torch.abs(pred-y)*1000
        weight = 1+2*(y/.02).clamp(0, 1)
        return (diff*weight).mean(), float(diff.mean().detach().cpu())

    histories, states = {}, {}
    for label, factory in (("spatial_unet", SpatialFlowUNet),
                           ("plain_spatial_cnn", PlainSpatialCNN),
                           ("local_1x1", LocalFlowMLP)):
        torch.manual_seed(3196)
        rng = np.random.default_rng(3196)
        model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.002)
        best = float("inf")
        best_epoch = 0
        curve = []
        for epoch in range(1, MAX_EPOCHS+1):
            model.train()
            order = rng.permutation(len(train_rows))
            losses = []
            for start in range(0, len(order), 12):
                batch = [train_rows[j] for j in order[start:start+12]]
                loss, _ = batch_loss(model, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                valid_mae = np.mean([batch_loss(model, valid_rows[i:i+12])[1]
                                     for i in range(0, len(valid_rows), 12)])
            curve.append({"epoch": epoch, "train_weighted_l1_mm": float(np.mean(losses)),
                          "validation_mae_mm": float(valid_mae)})
            if valid_mae < best:
                best, best_epoch = valid_mae, epoch
                states[label] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == 1:
                print(label, epoch, "train", round(float(np.mean(losses)), 3),
                      "val", round(float(valid_mae), 3), flush=True)
            if epoch-best_epoch >= PATIENCE:
                break
        histories[label] = {"best_epoch": best_epoch,
                            "best_validation_mae_mm": best, "curve": curve}
    torch.save({"states": states, "architecture": ["SpatialFlowUNet", "PlainSpatialCNN", "LocalFlowMLP"]},
               OUT / "pretrained.pt")
    (OUT / "training.json").write_text(json.dumps({"device": str(device),
        "loss": "weighted cell L1 on SFINCS next-depth; output volume normalized to current water plus 5 min rainfall",
        "selection": "lowest one-step MAE on geographic validation column 6, training rates only",
        "transitions": {"train": len(train_rows), "validation": len(valid_rows)},
        "models": histories}, indent=2)+"\n")
    print("Saved", OUT / "pretrained.pt")


def evaluate() -> None:
    import torch
    from flood_sfincs_physics_model import SpatialFlowUNet, PlainSpatialCNN, LocalFlowMLP

    torch.set_num_threads(4)
    with np.load(OUT / "sequences.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    states = torch.load(OUT / "pretrained.pt", map_location="cpu", weights_only=True)["states"]
    models = {}
    for label, factory in (("spatial_unet", SpatialFlowUNet),
                           ("plain_spatial_cnn", PlainSpatialCNN),
                           ("local_1x1", LocalFlowMLP)):
        model = factory()
        model.load_state_dict(states[label])
        models[label] = model.eval()
    buckets = {group: {label: [] for label in (*models, "persistence")}
               for group in ("heldout_seen_rates", "heldout_unseen_storm",
                             "validation_unseen_storm")}
    mass_errors = []
    for i in range(len(data["rate"])):
        split, rate = str(data["split"][i]), int(data["rate"][i])
        group = ("heldout_unseen_storm" if split == "test" and rate == 60 else
                 "heldout_seen_rates" if split == "test" else
                 "validation_unseen_storm" if split == "validation" and rate == 60 else None)
        if group is None:
            continue
        z = data["z"][i]
        true = data["h"][i]
        length = int(data["length"][i])
        increment = rate/1000*STEP_S/3600
        for label, model in (*models.items(), ("persistence", None)):
            prev = np.zeros_like(z)
            now = np.zeros_like(z)
            one_errors, roll_errors, wet_errors, interior_errors = [], [], [], []
            for step in range(1, length):
                x = _features(z, prev, now, rate)
                if model is None:
                    following = now+increment
                else:
                    with torch.no_grad():
                        following = model(torch.from_numpy(x[None]))[0, 0].numpy()
                target = true[step]
                roll_errors.append(float(np.mean(np.abs(following-target))*1000))
                interior_errors.append(float(np.mean(np.abs(
                    following[16:-16, 16:-16]-target[16:-16, 16:-16]))*1000))
                wet = target > .02
                if wet.any():
                    wet_errors.append(float(np.mean(np.abs(following[wet]-target[wet]))*1000))
                if model is not None:
                    true_x = _features(z, true[max(0, step-2)], true[step-1], rate)
                    with torch.no_grad():
                        one = model(torch.from_numpy(true_x[None]))[0, 0].numpy()
                    one_errors.append(float(np.mean(np.abs(one-target))*1000))
                    mass_errors.append(abs(float(following.mean()-step*increment))*1000)
                prev, now = now, following
            buckets[group][label].append({"final_rollout_mae_mm": roll_errors[-1],
                "final_interior_rollout_mae_mm": interior_errors[-1],
                "mean_rollout_mae_mm": float(np.mean(roll_errors)),
                "final_wet_gt20mm_mae_mm": wet_errors[-1] if wet_errors else None,
                "one_step_mae_mm": float(np.mean(one_errors)) if one_errors else None})
    summary = {}
    for group, arms in buckets.items():
        summary[group] = {}
        for label, rows in arms.items():
            summary[group][label] = {"cases": len(rows)}
            for key in rows[0]:
                vals = [r[key] for r in rows if r[key] is not None]
                summary[group][label][key] = float(np.mean(vals)) if vals else None
    result = {"target": "SFINCS synthetic final depth, not observed flood",
              "measurement": "each case counts equally; depth MAE over all 10m cells",
              "max_rollout_mass_balance_error_mean_depth_mm": max(mass_errors),
              "results": summary}
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


def plot() -> None:
    import matplotlib.pyplot as plt
    import torch
    from flood_sfincs_physics_model import SpatialFlowUNet, PlainSpatialCNN, LocalFlowMLP

    history = json.loads((OUT / "training.json").read_text())["models"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in (("spatial_unet", "#1261a0"),
                         ("plain_spatial_cnn", "#438c61"),
                         ("local_1x1", "#ae5724")):
        curve = history[label]["curve"]
        ax.plot([r["epoch"] for r in curve],
                [r["validation_mae_mm"] for r in curve],
                label=label.replace("_", " "), color=color)
        best = history[label]["best_epoch"]
        ax.scatter([best], [history[label]["best_validation_mae_mm"]],
                   color=color, s=25)
    ax.set(xlabel="Training epoch", ylabel="Held-out validation one-step MAE (mm)",
           title="SFINCS physics pretraining · separate LiDAR tiles")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "validation_curve.png", dpi=170)
    plt.close(fig)

    with np.load(OUT / "sequences.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    target_name = "r4c8_60mmh_20min"  # predetermined held-out geography and storm
    matches = np.where(data["name"] == target_name)[0]
    if len(matches) != 1:
        raise ValueError("Missing fixed held-out illustration")
    i = int(matches[0])
    states = torch.load(OUT / "pretrained.pt", map_location="cpu", weights_only=True)["states"]
    maps = {}
    for label, factory in (("spatial_unet", SpatialFlowUNet),
                           ("plain_spatial_cnn", PlainSpatialCNN),
                           ("local_1x1", LocalFlowMLP)):
        model = factory().eval()
        model.load_state_dict(states[label])
        prior = np.zeros((SIZE, SIZE), np.float32)
        now = prior.copy()
        for _ in range(1, int(data["length"][i])):
            x = _features(data["z"][i], prior, now, 60)
            with torch.no_grad():
                next_water = model(torch.from_numpy(x[None]))[0, 0].numpy()
            prior, now = now, next_water
        maps[label] = now*1000
    truth = data["h"][i, int(data["length"][i])-1]*1000
    cap = float(np.quantile(truth, .995))
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), constrained_layout=True)
    for ax, array, title in zip(axes, (truth, maps["spatial_unet"],
                                      maps["plain_spatial_cnn"],
                                      maps["local_1x1"],
                                      np.abs(maps["plain_spatial_cnn"]-truth)),
                                ("SFINCS target", "Spatial U-Net rollout",
                                 "Plain CNN rollout", "Local control rollout",
                                 "Plain CNN absolute error")):
        im = ax.imshow(array, cmap="Blues" if "error" not in title.lower() else "magma",
                       vmin=0, vmax=cap if "error" not in title.lower() else 50)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label="Depth (mm)" if "error" not in title.lower()
                     else "Error (mm)", shrink=.75)
    fig.suptitle("Held-out Bang Kapi tile r4c8 · unseen 60 mm/h rain for 20 min")
    fig.savefig(OUT / "heldout_example.png", dpi=170)
    plt.close(fig)
    print(OUT / "validation_curve.png", OUT / "heldout_example.png")


def plot_io() -> None:
    """Show every input and target for a fixed, actual training transition."""
    import matplotlib.pyplot as plt
    import torch
    from flood_sfincs_physics_model import PlainSpatialCNN

    with np.load(OUT / "sequences.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    matches = np.where(data["name"] == "r4c3_40mmh_15min")[0]
    if len(matches) != 1:
        raise ValueError("Missing fixed I/O example")
    i = int(matches[0])
    if data["split"][i] != "train" or int(data["length"][i]) != 4:
        raise ValueError("Unexpected I/O example split or times")
    z = data["z"][i]
    previous = data["h"][i, 1]  # t = 5 min
    current = data["h"][i, 2]  # t = 10 min
    target = data["h"][i, 3]  # t = 15 min
    rate = 40.
    x = _features(z, previous, current, rate)
    state = torch.load(OUT / "pretrained.pt", map_location="cpu",
                       weights_only=True)["states"]["plain_spatial_cnn"]
    model = PlainSpatialCNN().eval()
    model.load_state_dict(state)
    with torch.no_grad():
        prediction = model(torch.from_numpy(x[None]))[0, 0].numpy()
    error_mm = np.abs(prediction-target)*1000
    weight = 1+2*np.clip(target/.02, 0, 1)
    weighted_error = error_mm*weight
    loss_mm = float(weighted_error.mean())
    mae_mm = float(error_mm.mean())
    rain_increment_mm = rate*STEP_S/3600
    depth_cap = float(np.quantile(target, .995)*1000)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8), constrained_layout=True)
    layers = (
        (z-np.median(z), "1. Relative elevation", "m", "terrain", None),
        (z-gaussian_filter(z, 3), "2. Local relief", "m", "terrain", None),
        (previous*1000, "3. Previous depth · 5 min", "mm", "water", depth_cap),
        (current*1000, "4. Current depth · 10 min", "mm", "water", depth_cap),
        (None, "5. Rain during next 5 min", "", "rain", None),
        (target*1000, "6. SFINCS target · 15 min", "mm", "water", depth_cap),
        (prediction*1000, "7. CNN prediction · 15 min", "mm", "water", depth_cap),
        (weighted_error, "8. Weighted absolute error", "mm", "error", None),
    )
    for ax, (array, title, unit, kind, vmax) in zip(axes.flat, layers):
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if kind == "rain":
            ax.set_facecolor("#186d80")
            ax.text(.5, .57, f"{int(rate)} mm/hour", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=20,
                    fontweight="bold")
            ax.text(.5, .40, f"+{rain_increment_mm:.2f} mm per cell",
                    transform=ax.transAxes, ha="center", va="center",
                    color="white", fontsize=13)
            continue
        if kind == "terrain":
            limit = float(np.quantile(np.abs(array), .995))
            im = ax.imshow(array, cmap="RdBu_r", vmin=-limit, vmax=limit)
        elif kind == "water":
            im = ax.imshow(array, cmap="Blues", vmin=0, vmax=vmax)
        else:
            im = ax.imshow(array, cmap="magma", vmin=0,
                           vmax=float(np.quantile(array, .99)))
        fig.colorbar(im, ax=ax, label=unit, shrink=.72)
    fig.suptitle(
        "One actual pretraining example: five input layers → 48-channel dilated CNN → next depth\n"
        f"Bang Kapi 10 m tile r4c3 · weighted training loss {loss_mm:.2f} mm "
        f"· unweighted MAE {mae_mm:.2f} mm", fontsize=15)
    path = OUT / "training_io_example.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "collect", "train", "evaluate", "plot", "plot_io"))
    action = parser.parse_args().action
    {"prepare": prepare, "collect": collect, "train": train,
     "evaluate": evaluate, "plot": plot, "plot_io": plot_io}[action]()
