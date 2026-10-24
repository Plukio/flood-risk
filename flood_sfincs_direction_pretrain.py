"""Test spatial masking and a SFINCS velocity-direction loss.

Each sample maps a current terrain/water/surface state to next water depth and
the next SFINCS velocity vector. A paired arm sees uniform Manning n and the
other sees mapped WorldCover/road Manning n. Lat Krabang tiles are held out.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flood_sfincs_physics_model import water_projection
from flood_sfincs_physics_pretrain import OUT as OLD, _features
from flood_sfincs_surface_multiarea import OUT as MULTI, load_sequences
from flood_sfincs_surface_pretrain import (OUT as SURFACE, features as surface_features)


OUT = Path("artifacts/flood_bangkapi_surface/sfincs_direction_pretrain")
CHANNELS = 7  # previous/current depth + static 5 + one mask indicator
BATCH = 12
EPOCHS = 24
PATIENCE = 8
FLOW_WEIGHT = .01
DIRECTION_WEIGHT = 1.0


class FlowMaskCNN(nn.Module):
    """Dilated CNN with next-depth and two-component velocity heads."""
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(CHANNELS, 48, 3, padding=1)
        dilation = (1, 2, 4, 8, 16, 16, 8, 4, 2, 1)
        self.layers = nn.ModuleList(nn.Conv2d(48, 48, 3, padding=d,
                                               dilation=d) for d in dilation)
        self.depth_head = nn.Conv2d(48, 1, 1)
        self.velocity_head = nn.Conv2d(48, 2, 1)

    def forward(self, image):
        hidden = F.silu(self.entry(image))
        for layer in self.layers:
            hidden = F.silu(hidden+.2*layer(hidden))
        depth = water_projection(self.depth_head(hidden), image)
        velocity = self.velocity_head(hidden)
        return depth, velocity


def cases_for_area():
    # NpzFile is lazy: indexing it inside the case loop would repeatedly
    # decompress the entire archive, causing multi-gigabyte memory churn.
    with np.load(OLD / "sequences.npz") as archive:
        base = {key: archive[key] for key in archive.files}
    with np.load(SURFACE / "mapped_sequences.npz") as archive:
        bkp = {key: archive[key] for key in archive.files}
    multi = load_sequences()
    records = json.loads((MULTI / "manifest.json").read_text())
    arrays = []
    for i in range(len(base["rate"])):
        for arm, depth, uv, rough in (
            ("uniform", base["h"][i], bkp["uniform_uv"][i], np.full_like(base["z"][i], .035)),
            ("mapped", bkp["h"][i], bkp["uv"][i], bkp["manning"][i])):
            case_id = f"bang_kapi_{base['name'][i]}_{arm}"
            arrays.append({"region": "bang_kapi", "split": str(base["split"][i]),
                           "row": int(base["row"][i]), "col": int(base["col"][i]),
                           "rate": int(base["rate"][i]),
                           "length": int(base["length"][i]), "arm": arm,
                           "case_id": case_id,
                           "z": base["z"][i], "h": depth, "uv": uv,
                           "manning": rough})
    for i, record in enumerate(records):
        arrays.append({"region": record["region"],
                       "split": "test" if record["role"] == "untouched_test" else "train",
                       "row": record["row"], "col": record["col"],
                       "rate": record["rate"], "length": record["minutes"]//5+1,
                       "arm": record["arm"], "case_id": record["name"], "z": multi["z"][i],
                       "h": multi["h"][i], "uv": multi["uv"][i],
                       "manning": multi["manning"][i]})
    return arrays


def make_samples():
    samples = {"train": [], "validation": [], "test_bang_kapi": [],
               "test_lat_krabang": []}
    zero_n_cache = {}
    for case in cases_for_area():
        # The 60 mm/h x 20 min storm is the reserved unseen-forcing test.
        # The source generator also wrote it on a validation tile; never let
        # that copy leak into fitting.
        if case["region"] == "bang_kapi" and case["rate"] == 60 and case["split"] != "test":
            continue
        if case["region"] == "lat_krabang":
            group = "test_lat_krabang"
        elif case["split"] == "test":
            group = "test_bang_kapi"
        elif case["split"] == "validation" and case["rate"] != 60:
            group = "validation"
        else:
            group = "train"
        if case["arm"] == "uniform":
            n = np.full_like(case["z"], .035)
        else:
            n = case["manning"]
        for step in range(1, case["length"]):
            static = surface_features(case["z"], case["h"][max(0, step-2)],
                                      case["h"][step-1], case["rate"], n)
            # Keep the 6 standard channels together, append mask indicator.
            x = np.concatenate((static, np.zeros((1, 96, 96), np.float32)), axis=0)
            samples[group].append({"x": x, "depth": case["h"][step][None].copy(),
                                   "uv": case["uv"][step].copy(),
                                   "region": case["region"], "rate": case["rate"],
                                   "row": case["row"], "col": case["col"],
                                   "arm": case["arm"], "case_id": case["case_id"],
                                   "step": step})
    return samples


def initialize(model):
    checkpoint = torch.load(OLD / "pretrained.pt", map_location="cpu",
                            weights_only=True)["states"]["plain_spatial_cnn"]
    with torch.no_grad():
        model.entry.weight.zero_()
        model.entry.weight[:, :5].copy_(checkpoint["entry.weight"])
        model.entry.bias.copy_(checkpoint["entry.bias"])
        for name, layer in zip(("layers.0.weight", "layers.1.weight", "layers.2.weight",
                                "layers.3.weight", "layers.4.weight", "layers.5.weight",
                                "layers.6.weight", "layers.7.weight", "layers.8.weight",
                                "layers.9.weight"), model.layers):
            layer.weight.copy_(checkpoint[name])
            layer.bias.copy_(checkpoint[name.replace("weight", "bias")])
        model.depth_head.weight.copy_(checkpoint["head.weight"])
        model.depth_head.bias.copy_(checkpoint["head.bias"])
        nn.init.normal_(model.velocity_head.weight, mean=0., std=.002)
        nn.init.zeros_(model.velocity_head.bias)


def random_mask(x, rng):
    x = x.copy()
    mask = np.zeros((96, 96), np.float32)
    # Hide the older depth map. Keep current depth visible because the
    # conservation projection uses it to enforce the closed-domain water budget.
    for _ in range(1+int(rng.random() < .35)):
        size = int(rng.integers(8, 25))
        top = int(rng.integers(0, 97-size))
        left = int(rng.integers(0, 97-size))
        mask[top:top+size, left:left+size] = 1.
    x[2, mask > 0] = 0.
    x[6] = mask
    return x


def loss_fn(pred_depth, pred_uv, batch, direction_weight=1.0):
    depth = torch.from_numpy(np.stack([x["depth"] for x in batch])).to(pred_depth.device)
    uv = torch.from_numpy(np.stack([x["uv"] for x in batch])).to(pred_uv.device)
    depth_error = torch.abs(pred_depth-depth)*1000
    depth_weight = 1+2*(depth/.02).clamp(0, 1)
    depth_loss = (depth_error*depth_weight).mean()

    speed = torch.sqrt((uv**2).sum(dim=1, keepdim=True)+1e-12)
    pred_speed = torch.sqrt((pred_uv**2).sum(dim=1, keepdim=True)+1e-12)
    active = (speed > .01) & (depth > .005)
    mag_weight = (.05+speed/.05).clamp(max=1.) * active.float()
    component = F.smooth_l1_loss(pred_uv/.1, uv/.1, reduction="none").mean(dim=1, keepdim=True)
    magnitude_loss = ((component*1000*mag_weight).sum() /
                      mag_weight.sum().clamp_min(1.))
    dot = (pred_uv*uv).sum(dim=1, keepdim=True)
    cosine = dot / (pred_speed*speed).clamp_min(1e-6)
    direction_loss = ((1-cosine.clamp(-1, 1))*active.float()).sum() / \
                     active.float().sum().clamp_min(1.)
    return depth_loss+direction_weight*(FLOW_WEIGHT*magnitude_loss+
                                        DIRECTION_WEIGHT*direction_loss), \
        depth_loss, magnitude_loss, direction_loss, active


def train_variant(samples, label, mask_probability, direction_weight):
    torch.manual_seed(8803)
    shuffle_rng = np.random.default_rng(8803)
    augmentation_rng = np.random.default_rng(29017)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FlowMaskCNN().to(device)
    initialize(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00015, weight_decay=.002)
    best, best_epoch, best_state = float("inf"), 0, None
    curve = []
    train_rows, val_rows = samples["train"], samples["validation"]
    for epoch in range(1, EPOCHS+1):
        model.train()
        order = shuffle_rng.permutation(len(train_rows))
        train_losses = []
        for start in range(0, len(order), BATCH):
            batch = [train_rows[j] for j in order[start:start+BATCH]]
            xs = []
            for sample in batch:
                if augmentation_rng.random() < mask_probability:
                    xs.append(random_mask(sample["x"], augmentation_rng))
                else:
                    xs.append(sample["x"])
            x = torch.from_numpy(np.stack(xs)).to(device)
            pred_depth, pred_uv = model(x)
            loss, depth_loss, magnitude_loss, direction_loss, _ = loss_fn(
                pred_depth, pred_uv, batch, direction_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_losses.append([float(v.detach().cpu()) for v in
                                 (loss, depth_loss, magnitude_loss, direction_loss)])
        model.eval()
        val_losses, val_direction_count = [], 0
        with torch.no_grad():
            for start in range(0, len(val_rows), BATCH):
                batch = val_rows[start:start+BATCH]
                x = torch.from_numpy(np.stack([s["x"] for s in batch])).to(device)
                outputs = model(x)
                values = loss_fn(*outputs, batch, direction_weight)
                val_losses.append([float(v.detach().cpu()) for v in values[:4]])
                val_direction_count += int(values[4].sum().detach().cpu())
        train_mean = np.mean(train_losses, axis=0)
        val_mean = np.mean(val_losses, axis=0)
        curve.append({"epoch": epoch, "train_total": float(train_mean[0]),
                      "train_depth_mm": float(train_mean[1]),
                      "train_velocity_mm_equiv": float(train_mean[2]),
                      "train_direction_cosine_loss": float(train_mean[3]),
                      "validation_total": float(val_mean[0]),
                      "validation_depth_mm": float(val_mean[1]),
                      "validation_velocity_mm_equiv": float(val_mean[2]),
                      "validation_direction_cosine_loss": float(val_mean[3]),
                      "validation_active_flow_pixels": val_direction_count})
        print(label, "epoch", epoch, "train", round(float(train_mean[0]), 3),
              "val", round(float(val_mean[0]), 3), "depth", round(float(val_mean[1]), 3),
              "direction", round(float(val_mean[3]), 3), flush=True)
        if val_mean[0] < best:
            best, best_epoch = float(val_mean[0]), epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch-best_epoch >= PATIENCE:
            break
    dest = OUT / label
    dest.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "mask_probability": mask_probability},
               dest / "checkpoint.pt")
    (dest / "training.json").write_text(json.dumps({
        "best_epoch": best_epoch, "best_validation_total": best,
        "mask_probability": mask_probability, "curve": curve}, indent=2)+"\n")
    return dest / "checkpoint.pt"


def main():
    samples = make_samples()
    print({key: len(value) for key, value in samples.items()}, flush=True)
    a = train_variant(samples, "depth_only_control", 0., 0.)
    b = train_variant(samples, "direction_only", 0., 1.)
    c = train_variant(samples, "direction_plus_random_block_mask", .45, 1.)
    (OUT / "design.json").write_text(json.dumps({
        "teacher": "SFINCS v2.4.1 simulated 10 m depths and cell velocities",
        "input": "relative terrain, local relief, prior/current depth, rain, Manning n, mask indicator",
        "outputs": "next depth and next u/v velocity",
        "training_areas": "Bang Kapi and Bang Khae",
        "validation": "Bang Kapi held-out tiles, trained rain rates",
        "final_test": "Lat Krabang held-out tiles including an unseen 60 mm/h x 20 min storm",
        "mask": "random 8-24 cell contiguous blocks hide older depth input; current depth stays visible for exact closed-domain storage projection; mask channel marks missing prior-state cells",
        "flow_loss": "vector Huber plus cosine direction loss where target speed > 0.01 m/s and depth > 0.005 m",
        "loss_weights": {"depth": 1.0, "velocity_magnitude": FLOW_WEIGHT,
                         "velocity_direction": DIRECTION_WEIGHT},
        "checkpoints": {"depth_only": str(a), "direction_only": str(b),
                        "direction_plus_mask": str(c)},
        "caveat": "Simulator imitation, not measured flood accuracy."}, indent=2)+"\n")
    evaluate(samples)


def direction_metrics(pred, truth, depth):
    target = torch.from_numpy(truth).to(pred.device)
    d = torch.from_numpy(depth).to(pred.device)
    speed = torch.sqrt((target**2).sum(dim=1, keepdim=True)+1e-12)
    pred_speed = torch.sqrt((pred**2).sum(dim=1, keepdim=True)+1e-12)
    active = (speed > .01) & (d > .005)
    dot = (pred*target).sum(dim=1, keepdim=True)
    cosine = dot/(pred_speed*speed).clamp_min(1e-6)
    if not active.any():
        return {"active_pixels": 0, "direction_mae_deg": None,
                "within_30deg_fraction": None, "velocity_vector_mae_mps": None}
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))[active]
    vecerr = torch.sqrt(((pred-target)**2).sum(dim=1, keepdim=True))[active]
    return {"active_pixels": int(active.sum().cpu()),
            "direction_mae_deg": float(angle.mean().cpu()),
            "within_30deg_fraction": float((angle < 30).float().mean().cpu()),
            "velocity_vector_mae_mps": float(vecerr.mean().cpu())}


def evaluate(samples, checkpoint_dir=OUT, labels=None):
    torch.set_num_threads(4)
    device = torch.device("cpu")
    reports = {}
    if labels is None:
        labels = ("depth_only_control", "direction_only",
                  "direction_plus_random_block_mask")
    for label in labels:
        checkpoint = torch.load(checkpoint_dir/label/"checkpoint.pt", map_location="cpu",
                                weights_only=True)
        model = FlowMaskCNN().eval()
        model.load_state_dict(checkpoint["state"])
        results = {}
        for split in ("test_bang_kapi", "test_lat_krabang"):
            rows = samples[split]
            buckets = {}
            for start in range(0, len(rows), BATCH):
                batch = rows[start:start+BATCH]
                for masked_eval in (False, True):
                    rngs = [np.random.default_rng(5081 + start + j)
                            for j in range(len(batch))]
                    xs = [random_mask(s["x"], rngs[j]) if masked_eval else s["x"]
                          for j, s in enumerate(batch)]
                    x = torch.from_numpy(np.stack(xs)).to(device)
                    with torch.no_grad():
                        pred_depth, pred_uv = model(x)
                    for j, sample in enumerate(batch):
                        key = f"rate_{sample['rate']}_masked_{str(masked_eval).lower()}"
                        bucket = buckets.setdefault(key, {
                            "depth_mae": [], "wet_depth_mae": [], "angles": [],
                            "within_30": [], "vector_error": [], "active_pixels": 0})
                        truth_depth = sample["depth"]
                        error_mm = np.abs(pred_depth[j].cpu().numpy()-truth_depth)*1000
                        bucket["depth_mae"].append(float(error_mm.mean()))
                        wet = truth_depth > .02
                        if wet.any():
                            bucket["wet_depth_mae"].append(float(error_mm[wet].mean()))
                        if label not in ("depth_only_control", "paired_source",
                                         "paired_depth_control"):
                            metric = direction_metrics(
                                pred_uv[j:j+1], sample["uv"][None], truth_depth[None])
                            nactive = metric["active_pixels"]
                            if nactive:
                                bucket["angles"].append((metric["direction_mae_deg"], nactive))
                                bucket["within_30"].append((metric["within_30deg_fraction"], nactive))
                                bucket["vector_error"].append((metric["velocity_vector_mae_mps"], nactive))
                                bucket["active_pixels"] += nactive
            results[split] = {}
            for key, bucket in buckets.items():
                row = {"samples": len(bucket["depth_mae"]),
                       "next_step_depth_mae_mm": float(np.mean(bucket["depth_mae"])),
                       "wet_gt20mm_depth_mae_mm": (float(np.mean(bucket["wet_depth_mae"]))
                                                    if bucket["wet_depth_mae"] else None),
                       "active_flow_pixels": bucket["active_pixels"]}
                for source, target in (("angles", "direction_mae_deg"),
                                       ("within_30", "within_30deg_fraction"),
                                       ("vector_error", "velocity_vector_mae_mps")):
                    vals = bucket[source]
                    row[target] = (float(sum(a*w for a, w in vals)/sum(w for _, w in vals))
                                   if vals else None)
                results[split][key] = row
        reports[label] = results
    result = {
        "target": "SFINCS simulated depth and velocity, not observed flood outcomes",
        "evaluation": "tile holdouts; each storm rate is grouped per sample; each model is tested with complete inputs and with random missing blocks in the older depth map",
        "direction_metric_mask": "target speed > 0.01 m/s and target depth > 0.005 m",
        "wet_depth_metric_mask": "target depth > 0.02 m",
        "results": reports}
    (checkpoint_dir / "evaluation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


def evaluate_rollouts(checkpoint_dir=OUT, labels=None):
    """Test complete 15/20-minute scenarios using the model's own depths."""
    torch.set_num_threads(4)
    cases = [case for case in cases_for_area()
             if case["region"] == "lat_krabang" or
             (case["region"] == "bang_kapi" and case["split"] == "test")]
    reports = {}
    if labels is None:
        labels = ("depth_only_control", "direction_only",
                  "direction_plus_random_block_mask")
    case_rows = {}
    for label in labels:
        state = torch.load(checkpoint_dir/label/"checkpoint.pt", map_location="cpu",
                           weights_only=True)["state"]
        model = FlowMaskCNN().eval()
        model.load_state_dict(state)
        buckets = {}
        case_rows[label] = []
        for region in ("bang_kapi", "lat_krabang"):
            for rate in (20, 40, 80, 60):
                subset = [case for case in cases if case["region"] == region
                          and case["rate"] == rate]
                for start in range(0, len(subset), BATCH):
                    batch = subset[start:start+BATCH]
                    lengths = {case["length"] for case in batch}
                    if len(lengths) != 1:
                        raise ValueError("Mixed scenario lengths in rollout batch")
                    previous = np.zeros((len(batch), 96, 96), np.float32)
                    current = previous.copy()
                    for step in range(1, batch[0]["length"]):
                        xs = []
                        for j, case in enumerate(batch):
                            x = surface_features(case["z"], previous[j], current[j],
                                                 case["rate"], case["manning"])
                            xs.append(np.concatenate((x, np.zeros((1, 96, 96),
                                                                    np.float32)), axis=0))
                        with torch.no_grad():
                            pred_depth, pred_uv = model(torch.from_numpy(np.stack(xs)))
                        previous, current = current, pred_depth[:, 0].cpu().numpy()
                    for j, case in enumerate(batch):
                        truth = case["h"][case["length"]-1]
                        error_mm = np.abs(current[j]-truth)*1000
                        wet = truth > .02
                        case_rows[label].append({
                            "case_id": case["case_id"], "region": case["region"],
                            "row": case["row"], "col": case["col"],
                            "rate": rate, "arm": case["arm"],
                            "final_depth_mae_mm": float(error_mm.mean()),
                            "final_wet_gt20mm_mae_mm": (float(error_mm[wet].mean())
                                                        if wet.any() else None)})
                        metric = (direction_metrics(pred_uv[j:j+1],
                                  case["uv"][case["length"]-1][None],
                                  truth[None, None])
                                  if label not in ("depth_only_control", "paired_source",
                                                   "paired_depth_control")
                                  else None)
                        for key in (f"{region}_rate_{rate}",
                                    f"{region}_rate_{rate}_{case['arm']}"):
                            bucket = buckets.setdefault(key, {
                                "mae": [], "wet_mae": [], "angles": [],
                                "within_30": [], "vector_error": [], "active_pixels": 0})
                            bucket["mae"].append(float(error_mm.mean()))
                            if wet.any():
                                bucket["wet_mae"].append(float(error_mm[wet].mean()))
                            if metric and metric["active_pixels"]:
                                count = metric["active_pixels"]
                                bucket["angles"].append((metric["direction_mae_deg"], count))
                                bucket["within_30"].append((metric["within_30deg_fraction"], count))
                                bucket["vector_error"].append((metric["velocity_vector_mae_mps"], count))
                                bucket["active_pixels"] += count
        report = {}
        for key, bucket in buckets.items():
            row = {"cases": len(bucket["mae"]),
                   "final_depth_mae_mm": float(np.mean(bucket["mae"])),
                   "final_wet_gt20mm_mae_mm": (float(np.mean(bucket["wet_mae"]))
                                                 if bucket["wet_mae"] else None),
                   "active_flow_pixels": bucket["active_pixels"]}
            for source, target in (("angles", "final_direction_mae_deg"),
                                   ("within_30", "final_within_30deg_fraction"),
                                   ("vector_error", "final_velocity_vector_mae_mps")):
                vals = bucket[source]
                row[target] = (float(sum(v*n for v, n in vals)/sum(n for _, n in vals))
                               if vals else None)
            report[key] = row
        reports[label] = report
    result = {"target": "SFINCS simulated final depth and velocity",
              "evaluation": "held-out 960 m tiles; autoregressive 5-minute steps for complete 15/20-minute storms",
              "results": reports}
    (checkpoint_dir / "rollout_evaluation.json").write_text(json.dumps(result, indent=2)+"\n")
    (checkpoint_dir / "rollout_case_rows.json").write_text(
        json.dumps(case_rows, indent=2)+"\n")
    print(json.dumps(result, indent=2))


def plot_training_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = (("depth_only_control", "Depth only"),
              ("direction_only", "Depth + direction"),
              ("direction_plus_random_block_mask", "Depth + direction + mask"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for key, name in labels:
        result = json.loads((OUT/key/"training.json").read_text())
        epochs = [row["epoch"] for row in result["curve"]]
        depth = [row["validation_depth_mm"] for row in result["curve"]]
        axes[0].plot(epochs, depth, label=name)
        if key != "depth_only_control":
            direction = [row["validation_direction_cosine_loss"]
                         for row in result["curve"]]
            axes[1].plot(epochs, direction, label=name)
    axes[0].set(title="Bang Kapi validation: next-depth loss",
                xlabel="Epoch", ylabel="Depth-weighted absolute error (mm)")
    axes[1].set(title="Bang Kapi validation: simulated flow direction",
                xlabel="Epoch", ylabel="Mean 1 − cos(angle) on moving wet cells")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.savefig(OUT/"training_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
