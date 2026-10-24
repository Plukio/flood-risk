"""Add velocity supervision and prior-state masking to the strongest rollout CNN.

The starting checkpoint was fitted on paired mapped/uniform SFINCS rollouts.
All three continuation arms use that same checkpoint, Bang Kapi and Bang Khae
training pairs, and the same paired rollout loss. Lat Krabang stays held out.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flood_sfincs_direction_pretrain import (
    DIRECTION_WEIGHT, FLOW_WEIGHT, FlowMaskCNN, evaluate, evaluate_rollouts,
    make_samples, random_mask,
)
from flood_sfincs_surface_multiarea import OUT as MULTI, load_sequences
from flood_sfincs_surface_pretrain import features, load_data


OUT = Path("artifacts/flood_bangkapi_surface/sfincs_direction_pretrain/paired_extension")
START = MULTI / "paired_fit" / "pretrained.pt"
EPOCHS = 12
BATCH_PAIRS = 4
PAIR_WEIGHT = 2.0
LABELS = ("paired_source", "paired_depth_control", "paired_direction",
          "paired_direction_mask")


def load_pairs():
    base, mapped = load_data()
    training, validation = [], []
    zeros = np.zeros((96, 96), np.float32)
    for i in range(len(base["rate"])):
        split, rate = str(base["split"][i]), int(base["rate"][i])
        if split not in ("train", "validation") or rate == 60:
            continue
        z = base["z"][i]
        n0 = np.full_like(z, .035)
        n1 = mapped["manning"][i]
        row = {
            "x": np.stack((features(z, zeros, zeros, rate, n0),
                           features(z, zeros, zeros, rate, n1))),
            "h": np.stack((base["h"][i], mapped["h"][i])),
            "uv": np.stack((mapped["uniform_uv"][i], mapped["uv"][i])),
        }
        (training if split == "train" else validation).append(row)

    extra = load_sequences()
    records = json.loads((MULTI / "manifest.json").read_text())
    lookup = {(r["region"], r["row"], r["col"], r["rate"], r["minutes"],
               r["arm"]): i for i, r in enumerate(records)}
    keys = sorted({(r["region"], r["row"], r["col"], r["rate"], r["minutes"])
                   for r in records if r["region"] == "bang_khae"})
    for key in keys:
        i0, i1 = lookup[(*key, "uniform")], lookup[(*key, "mapped")]
        z = extra["z"][i0]
        training.append({
            "x": np.stack((features(z, zeros, zeros, key[3], extra["manning"][i0]),
                           features(z, zeros, zeros, key[3], extra["manning"][i1]))),
            "h": np.stack((extra["h"][i0], extra["h"][i1])),
            "uv": np.stack((extra["uv"][i0], extra["uv"][i1])),
        })
    if len(training) != 222 or len(validation) != 27:
        raise ValueError(f"Unexpected pair counts: {len(training)}, {len(validation)}")
    return training, validation


def initialize_surface(model):
    state = torch.load(START, map_location="cpu", weights_only=True)["state"]
    with torch.no_grad():
        model.entry.weight.zero_()
        model.entry.weight[:, :6].copy_(state["entry.weight"])
        model.entry.bias.copy_(state["entry.bias"])
        for i, layer in enumerate(model.layers):
            layer.weight.copy_(state[f"layers.{i}.weight"])
            layer.bias.copy_(state[f"layers.{i}.bias"])
        model.depth_head.weight.copy_(state["head.weight"])
        model.depth_head.bias.copy_(state["head.bias"])
        torch.nn.init.normal_(model.velocity_head.weight, 0, .002)
        torch.nn.init.zeros_(model.velocity_head.bias)


def batch_metrics(model, batch, device, direction_weight, mask_probability=0., rng=None):
    static = torch.from_numpy(np.concatenate([r["x"] for r in batch])).to(device)
    target_h = torch.from_numpy(np.concatenate([r["h"] for r in batch])).to(device)
    target_uv = torch.from_numpy(np.concatenate([r["uv"] for r in batch])).to(device)
    previous = torch.zeros_like(target_h[:, :1])
    current = previous.clone()
    depth_sum = delta_sum = magnitude_sum = direction_sum = 0.
    for step in (1, 2, 3):
        mask = torch.zeros_like(previous)
        if step == 3 and mask_probability:
            for pair_index in range(len(batch)):
                if rng.random() < mask_probability:
                    drawn = random_mask(np.zeros((7, 96, 96), np.float32), rng)[6]
                    mask[2*pair_index:2*pair_index+2, 0] = torch.from_numpy(drawn).to(device)
        frame = torch.cat((static[:, :2], previous/.05*(1-mask),
                           current/.05, static[:, 4:], mask), dim=1)
        following, velocity = model(frame)
        truth = target_h[:, step:step+1]
        weight = 1+2*(truth/.02).clamp(0, 1)
        depth = (weight*torch.abs(following-truth)*1000).mean()
        true_delta = truth[1::2]-truth[::2]
        predicted_delta = following[1::2]-following[::2]
        delta_weight = 1+4*(true_delta.abs()/.005).clamp(0, 1)
        delta = (delta_weight*torch.abs(predicted_delta-true_delta)*1000).mean()

        uv = target_uv[:, step]
        speed = torch.sqrt((uv**2).sum(1, keepdim=True)+1e-12)
        predicted_speed = torch.sqrt((velocity**2).sum(1, keepdim=True)+1e-12)
        active = (speed > .01) & (truth > .005)
        mag_weight = (.05+speed/.05).clamp(max=1.)*active.float()
        component = F.smooth_l1_loss(velocity/.1, uv/.1,
                                     reduction="none").mean(1, keepdim=True)
        magnitude = (component*1000*mag_weight).sum()/mag_weight.sum().clamp_min(1.)
        cosine = (velocity*uv).sum(1, keepdim=True)/(
            predicted_speed*speed).clamp_min(1e-6)
        direction = ((1-cosine.clamp(-1, 1))*active.float()).sum()/(
            active.float().sum().clamp_min(1.))
        depth_sum += depth/3
        delta_sum += delta/3
        magnitude_sum += magnitude/3
        direction_sum += direction/3
        previous, current = current, following
    paired = depth_sum+PAIR_WEIGHT*delta_sum
    total = paired+direction_weight*(FLOW_WEIGHT*magnitude_sum+
                                     DIRECTION_WEIGHT*direction_sum)
    return total, paired, depth_sum, delta_sum, magnitude_sum, direction_sum


def train_variant(training, validation, label, direction_weight, mask_probability):
    torch.manual_seed(44071)
    shuffle_rng = np.random.default_rng(44071)
    mask_rng = np.random.default_rng(8161)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FlowMaskCNN().to(device)
    initialize_surface(model)
    velocity_params = list(model.velocity_head.parameters())
    trunk_params = [param for name, param in model.named_parameters()
                    if not name.startswith("velocity_head.")]
    optimizer = torch.optim.AdamW([
        {"params": trunk_params, "lr": .00005},
        {"params": velocity_params, "lr": .0002}], weight_decay=.002)
    best, best_epoch, best_state = float("inf"), 0, None
    curve = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        order = shuffle_rng.permutation(len(training))
        train_values = []
        for start in range(0, len(order), BATCH_PAIRS):
            batch = [training[j] for j in order[start:start+BATCH_PAIRS]]
            metrics = batch_metrics(model, batch, device, direction_weight,
                                    mask_probability, mask_rng)
            optimizer.zero_grad()
            metrics[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_values.append([float(x.detach().cpu()) for x in metrics])
        model.eval()
        with torch.no_grad():
            val_values = [batch_metrics(model, validation[i:i+BATCH_PAIRS],
                                        device, direction_weight)
                          for i in range(0, len(validation), BATCH_PAIRS)]
        tr = np.mean(train_values, axis=0)
        val = np.mean([[float(x.detach().cpu()) for x in row]
                       for row in val_values], axis=0)
        curve.append({"epoch": epoch,
                      "train_total": float(tr[0]), "train_paired_depth_delta": float(tr[1]),
                      "validation_total": float(val[0]),
                      "validation_paired_depth_delta": float(val[1]),
                      "validation_depth_mm": float(val[2]),
                      "validation_delta_mm": float(val[3]),
                      "validation_velocity_scaled": float(val[4]),
                      "validation_direction_cosine": float(val[5])})
        print(label, "epoch", epoch, "val_pair", round(float(val[1]), 3),
              "val_depth", round(float(val[2]), 3),
              "val_direction", round(float(val[5]), 3), flush=True)
        # Select every arm on the operational depth + pair-difference target.
        if val[1] < best:
            best, best_epoch = float(val[1]), epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    dest = OUT/label
    dest.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "start": str(START),
                "best_epoch": best_epoch}, dest/"checkpoint.pt")
    (dest/"training.json").write_text(json.dumps({
        "start": str(START), "best_epoch": best_epoch,
        "best_validation_paired_depth_delta": best,
        "mask_probability_step3": mask_probability,
        "direction_weight": direction_weight, "curve": curve}, indent=2)+"\n")


def tile_bootstrap_report():
    case_rows = json.loads((OUT/"rollout_case_rows.json").read_text())
    rng = np.random.default_rng(4711)
    results = {}
    for region in ("bang_kapi", "lat_krabang"):
        for storm, rates in (("trained_rates", {20, 40, 80}),
                             ("unseen_60", {60})):
            tile_means = {}
            for label, rows in case_rows.items():
                grouped = defaultdict(list)
                for row in rows:
                    if row["region"] == region and row["rate"] in rates:
                        grouped[(row["row"], row["col"])].append(
                            row["final_depth_mae_mm"])
                tile_means[label] = {key: float(np.mean(values))
                                     for key, values in grouped.items()}
            keys = sorted(tile_means["paired_depth_control"])
            if not keys or any(set(values) != set(keys)
                               for values in tile_means.values()):
                raise ValueError(f"Unpaired tile results for {region}, {storm}")
            comparisons = {}
            for reference, candidate in (
                ("paired_depth_control", "paired_direction"),
                ("paired_depth_control", "paired_direction_mask"),
                ("paired_direction", "paired_direction_mask")):
                differences = np.array([tile_means[candidate][key]-
                                        tile_means[reference][key] for key in keys])
                sampled = differences[rng.integers(0, len(keys),
                                                  (20000, len(keys)))].mean(1)
                comparisons[f"{candidate}_minus_{reference}"] = {
                    "mean_difference_mm": float(differences.mean()),
                    "median_difference_mm": float(np.median(differences)),
                    "tiles_improved": int((differences < 0).sum()),
                    "tile_bootstrap_95pct_mm": [float(v) for v in
                                                 np.quantile(sampled, [.025, .975])]}
            results[f"{region}_{storm}"] = {"tiles": len(keys),
                                              "comparisons": comparisons}
    report = {
        "unit": "mean final-depth MAE difference per 960 m tile; negative favors candidate",
        "resampling": "20,000 bootstrap resamples of geographic tiles, seed 4711; each tile averages its uniform and mapped roughness cases and all selected rain rates",
        "caveat": "Tiles may be spatially correlated; this is synthetic simulator agreement, not observed flood skill",
        "results": results}
    (OUT/"tile_bootstrap.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def plot_training_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for label, name in (("paired_depth_control", "Depth only"),
                        ("paired_direction", "Depth + direction"),
                        ("paired_direction_mask", "Depth + direction + mask")):
        data = json.loads((OUT/label/"training.json").read_text())
        epochs = [row["epoch"] for row in data["curve"]]
        axes[0].plot(epochs,
                     [row["validation_paired_depth_delta"] for row in data["curve"]],
                     label=name)
        if label != "paired_depth_control":
            axes[1].plot(epochs,
                         [row["validation_direction_cosine"] for row in data["curve"]],
                         label=name)
    source = json.loads((START.parent/"training.json").read_text())
    source_row = next(row for row in source["curve"]
                      if row["epoch"] == source["best_epoch"])
    axes[0].axhline(source_row["validation_total_mm"], color="0.4",
                    linestyle="--", label="Corrected paired source")
    axes[0].set(xlabel="Continuation epoch", ylabel="Paired depth + difference loss (mm)",
                title="Bang Kapi geographic validation")
    axes[1].set(xlabel="Continuation epoch", ylabel="Mean 1 − cos(angle)",
                title="Simulated direction on moving wet cells")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.savefig(OUT/"paired_training_curves.png", dpi=160)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    training, validation = load_pairs()
    print("paired training", len(training), "validation", len(validation), flush=True)
    torch.manual_seed(44071)
    source = FlowMaskCNN()
    initialize_surface(source)
    source_dir = OUT/LABELS[0]
    source_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": source.state_dict(), "start": str(START),
                "best_epoch": 0}, source_dir/"checkpoint.pt")
    train_variant(training, validation, LABELS[1], 0., 0.)
    train_variant(training, validation, LABELS[2], 1., 0.)
    train_variant(training, validation, LABELS[3], 1., .45)
    (OUT/"design.json").write_text(json.dumps({
        "teacher": "SFINCS v2.4.1 simulated depth and velocity",
        "start": str(START),
        "training": "Bang Kapi and Bang Khae paired uniform/mapped 15-minute scenarios",
        "validation": "Bang Kapi geographic validation tiles at 20, 40, 80 mm/h",
        "test": "Bang Kapi held-out tiles and all Lat Krabang tiles, including unseen 60 mm/h for 20 minutes",
        "selection": "minimum unmasked validation paired depth plus 2 x roughness-difference loss",
        "mask": "at step 3 only, hides older depth on the same 80-240 m random blocks for each uniform/mapped pair; current depth remains visible for water storage",
        "flow_loss": "0.01 x normalized velocity Huber plus 1 x cosine direction where speed > 0.01 m/s and depth > 0.005 m",
        "caveat": "simulator imitation; no verified flood extent or urban drainage"}, indent=2)+"\n")
    samples = make_samples()
    evaluate(samples, checkpoint_dir=OUT, labels=LABELS)
    evaluate_rollouts(checkpoint_dir=OUT, labels=LABELS)
    tile_bootstrap_report()
    plot_training_curves()


if __name__ == "__main__":
    main()
