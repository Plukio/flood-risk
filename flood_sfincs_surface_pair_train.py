"""Fine-tune surface CNN on paired SFINCS rollouts from the same dry start.

The paired target is the simulated mapped-minus-uniform depth difference.
Both arms use the same terrain and rain, so the pair isolates the *assumed*
surface roughness scenario. Lat Krabang remains outside all weight fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flood_sfincs_surface_multiarea import OUT as MULTI, load_sequences
from flood_sfincs_surface_pretrain import (SurfaceSpatialCNN, features,
                                           load_data)


OUT = MULTI / "paired_fit"
EPOCHS = 12
PAIR_WEIGHT = 2.


def pairs():
    base, mapped = load_data()
    train_pairs, valid_pairs = [], []
    zeros = np.zeros((96, 96), np.float32)
    for i in range(len(base["rate"])):
        split, rate = str(base["split"][i]), int(base["rate"][i])
        if split not in ("train", "validation") or rate == 60:
            continue
        z = base["z"][i]
        n0 = np.full_like(z, .035)
        n1 = mapped["manning"][i]
        static = np.stack((features(z, zeros, zeros, rate, n0),
                           features(z, zeros, zeros, rate, n1)))
        target = np.stack((base["h"][i], mapped["h"][i]))
        (train_pairs if split == "train" else valid_pairs).append((static, target))

    extra = load_sequences()
    records = json.loads((MULTI / "manifest.json").read_text())
    index = {(r["region"], r["row"], r["col"], r["rate"], r["minutes"], r["arm"]): i
             for i, r in enumerate(records)}
    keys = sorted({(r["region"], r["row"], r["col"], r["rate"], r["minutes"])
                   for r in records if r["region"] == "bang_khae"})
    for key in keys:
        i0, i1 = index[(*key, "uniform")], index[(*key, "mapped")]
        z = extra["z"][i0]
        n0, n1 = extra["manning"][i0], extra["manning"][i1]
        static = np.stack((features(z, zeros, zeros, key[3], n0),
                           features(z, zeros, zeros, key[3], n1)))
        target = np.stack((extra["h"][i0], extra["h"][i1]))
        train_pairs.append((static, target))
    return train_pairs, valid_pairs


def batch_metrics(model, batch, device, augmentation=False, rng=None):
    x = torch.from_numpy(np.concatenate([r[0] for r in batch])).to(device)
    target = torch.from_numpy(np.concatenate([r[1] for r in batch])).to(device)
    if augmentation:
        k = int(rng.integers(4))
        x, target = torch.rot90(x, k, (-2, -1)), torch.rot90(target, k, (-2, -1))
        if rng.random() < .5:
            x, target = torch.flip(x, (-1,)), torch.flip(target, (-1,))
    previous = torch.zeros_like(target[:, :1])
    current = previous.clone()
    total_depth = total_delta = 0.
    for step in (1, 2, 3):
        frame = torch.cat((x[:, :2], previous/.05, current/.05, x[:, 4:]), dim=1)
        following = model(frame)
        truth = target[:, step:step+1]
        weight = 1+2*(truth/.02).clamp(0, 1)
        depth_loss = (weight*torch.abs(following-truth)*1000).mean()
        true_delta = truth[1::2]-truth[::2]
        predicted_delta = following[1::2]-following[::2]
        delta_weight = 1+4*(torch.abs(true_delta)/.005).clamp(0, 1)
        delta_loss = (delta_weight*torch.abs(predicted_delta-true_delta)*1000).mean()
        total_depth = total_depth+depth_loss/3
        total_delta = total_delta+delta_loss/3
        previous, current = current, following
    return total_depth+PAIR_WEIGHT*total_delta, total_depth, total_delta


def main():
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    training, validation = pairs()
    model = SurfaceSpatialCNN().to(device)
    start = MULTI / "expanded_fit" / "pretrained.pt"
    model.load_state_dict(torch.load(start, map_location="cpu",
                                     weights_only=True)["state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.002)
    rng = np.random.default_rng(421)
    best = float("inf")
    best_state = None
    best_epoch = 0
    curve = []
    print(f"device={device}; paired train={len(training)}, validation={len(validation)}", flush=True)
    for epoch in range(1, EPOCHS+1):
        model.train()
        order = rng.permutation(len(training))
        train_values = []
        for start in range(0, len(order), 4):
            batch = [training[j] for j in order[start:start+4]]
            loss, depth, delta = batch_metrics(model, batch, device, True, rng)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_values.append((float(loss.detach().cpu()),
                                 float(depth.detach().cpu()),
                                 float(delta.detach().cpu())))
        model.eval()
        with torch.no_grad():
            scores = [batch_metrics(model, validation[j:j+4], device)
                      for j in range(0, len(validation), 4)]
            val = np.mean([[float(x.detach().cpu()) for x in score]
                           for score in scores], axis=0)
        tr = np.mean(train_values, axis=0)
        curve.append({"epoch": epoch, "train_total_mm": float(tr[0]),
                      "train_depth_mm": float(tr[1]), "train_delta_mm": float(tr[2]),
                      "validation_total_mm": float(val[0]),
                      "validation_depth_mm": float(val[1]),
                      "validation_delta_mm": float(val[2])})
        print(f"epoch={epoch} train={tr[0]:.3f} val={val[0]:.3f} "
              f"val_depth={val[1]:.3f} val_delta={val[2]:.3f}", flush=True)
        if val[0] < best:
            best, best_epoch = float(val[0]), epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "architecture": "SurfaceSpatialCNN",
                "sixth_channel": "Manning n / 0.035 - 1"}, OUT / "pretrained.pt")
    (OUT / "training.json").write_text(json.dumps({
        "target": "paired SFINCS mapped-minus-uniform 5/10/15-minute depth differences",
        "initialisation": str(MULTI / "expanded_fit" / "pretrained.pt"),
        "pair_weight": PAIR_WEIGHT,
        "delta_cell_weight": "1 + 4*clip(abs(true delta)/0.005 m, 0, 1)",
        "selection": "minimum combined rollout loss on Bang Kapi geographic validation tiles",
        "train_pairs": len(training), "validation_pairs": len(validation),
        "best_epoch": best_epoch, "curve": curve}, indent=2)+"\n")
    print(f"Saved best epoch {best_epoch}, validation total {best:.3f} mm", flush=True)


if __name__ == "__main__":
    main()
