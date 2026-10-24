"""Add Bang Khae to surface-flow pretraining; reserve Lat Krabang for transfer.

Targets are SFINCS synthetic depth maps, not observed inundation. The only
surface parameter changed is mapped, assumed Manning roughness. All cases use
closed 960 m tiles and no drains, pumps, infiltration or building barriers.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import torch
from netCDF4 import Dataset
from rasterio.windows import Window

from flood_sfincs_lidar_trial import IMAGE
from flood_sfincs_physics_model import PlainSpatialCNN
from flood_sfincs_physics_pretrain import (OUT as OLD, SIZE, STEP_S, _features)
from flood_sfincs_surface_pretrain import (OUT as BASE, SurfaceSpatialCNN,
                                           features, train)
from flood_sfincs_surface_probe import static_layers


OUT = Path("artifacts/flood_bangkapi_surface/sfincs_surface_multiarea")
REGIONS = {
    "bang_khae": {
        "dtm": Path("data/flood_bangkok_transfer/dtm_1040_10m.tif"),
        "tiles": [(r, c) for r in (0, 2, 4, 6, 8) for c in (0, 2, 4, 6)],
        "storms": [(20, 15), (40, 15), (80, 15)],
        "role": "extra_train"},
    "lat_krabang": {
        "dtm": Path("data/flood_sfincs_transfer/dtm_lat_krabang_10m.tif"),
        "tiles": [(r, c) for r in (1, 3, 5, 7) for c in (0, 2, 4, 6, 8)],
        "storms": [(20, 15), (40, 15), (80, 15), (60, 20)],
        "role": "untouched_test"},
}


def static():
    for region, spec in REGIONS.items():
        dest = OUT / "regions" / region
        dest.mkdir(parents=True, exist_ok=True)
        cover, road, manning, audit = static_layers(spec["dtm"])
        np.savez_compressed(dest / "surface_10m.npz", worldcover=cover,
                            road_corridor=road, assumed_manning_n=manning)
        audit.update({"region": region, "role": spec["role"],
                      "dtm": str(spec["dtm"]), "selected_tiles": spec["tiles"]})
        (dest / "surface_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
        print(region, "grid", cover.shape, "road cells", audit["road_touched_cells"],
              "tiles", len(spec["tiles"]), flush=True)


def manifest():
    rows = []
    for region, spec in REGIONS.items():
        for row, col in spec["tiles"]:
            for rate, minutes in spec["storms"]:
                for arm in ("uniform", "mapped"):
                    name = f"{region}_r{row}c{col}_{rate}mmh_{minutes}min_{arm}"
                    rows.append({"name": name, "region": region,
                                 "row": row, "col": col, "rate": rate,
                                 "minutes": minutes, "arm": arm,
                                 "role": spec["role"]})
    return rows


def prepare():
    rows = manifest()
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    for region, spec in REGIONS.items():
        with rasterio.open(spec["dtm"]) as dtm:
            if dtm.crs.to_epsg() != 32647 or dtm.res != (10., 10.):
                raise ValueError(f"Unexpected terrain grid in {region}")
            with np.load(OUT / "regions" / region / "surface_10m.npz") as d:
                manning = d["assumed_manning_n"]
            if dtm.shape != manning.shape:
                raise ValueError(f"Surface and terrain misaligned for {region}")
            for record in (x for x in rows if x["region"] == region):
                row, col = record["row"], record["col"]
                window = Window(col*SIZE, row*SIZE, SIZE, SIZE)
                z = dtm.read(1, window=window).astype(np.float32)
                n = manning[row*SIZE:(row+1)*SIZE, col*SIZE:(col+1)*SIZE]
                if z.shape != (SIZE, SIZE) or n.shape != z.shape or not np.isfinite(z).all():
                    raise ValueError(f"Invalid tile {record['name']}")
                t = dtm.window_transform(window)
                x0, y0 = t.c, t.f-SIZE*10
                rate, minutes = record["rate"], record["minutes"]
                duration_s = minutes*60
                end = datetime(2026, 10, 23)+timedelta(seconds=duration_s)
                dest = OUT / "cases" / record["name"]
                dest.mkdir(parents=True, exist_ok=True)
                np.savetxt(dest / "sfincs.dep", np.flipud(z), fmt="%.8f")
                np.savetxt(dest / "sfincs.msk", np.ones_like(z, np.int8), fmt="%d")
                (dest / "sfincs.prcp").write_text(f"0 {rate}\n{duration_s} {rate}\n")
                roughness = "manning = 0.035\n"
                if record["arm"] == "mapped":
                    np.flipud(n).T.astype("<f4").tofile(dest / "sfincs.man")
                    roughness = "manningfile = sfincs.man\n"
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
{roughness}qinf = 0
dtmax = 2.5
dtout = {STEP_S}
dtmaxout = {duration_s}
storecumprcp = 1
storevel = 1
""")
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2)+"\n")
    (OUT / "case_names.txt").write_text("\n".join(x["name"] for x in rows)+"\n")
    (OUT / "case_names_mapped.txt").write_text(
        "\n".join(x["name"] for x in rows if x["arm"] == "mapped")+"\n")
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
    (OUT / "run_mapped.sh").write_text((OUT / "run_all.sh").read_text().replace(
        "case_names.txt", "case_names_mapped.txt"))
    print(f"Prepared {len(rows)} cases across two additional areas")


def collect():
    rows = json.loads((OUT / "manifest.json").read_text())
    zs, ns, hs, uvs = [], [], [], []
    max_n_error = max_balance_mm = 0.
    terrain_cache = {}
    roughness_cache = {}
    for record in rows:
        region, row, col = record["region"], record["row"], record["col"]
        if region not in terrain_cache:
            with rasterio.open(REGIONS[region]["dtm"]) as dtm:
                terrain_cache[region] = dtm.read(1)
            with np.load(OUT / "regions" / region / "surface_10m.npz") as d:
                roughness_cache[region] = d["assumed_manning_n"]
        z = terrain_cache[region][row*SIZE:(row+1)*SIZE, col*SIZE:(col+1)*SIZE]
        n = (roughness_cache[region][row*SIZE:(row+1)*SIZE, col*SIZE:(col+1)*SIZE]
             if record["arm"] == "mapped" else np.full_like(z, .035))
        with Dataset(OUT / "cases" / record["name"] / "sfincs_map.nc") as ds:
            h = np.asarray(ds.variables["h"][:], np.float32)[:, ::-1, :]
            if "u" not in ds.variables or "v" not in ds.variables:
                raise ValueError(f"Missing saved velocity in {record['name']}")
            u = np.asarray(ds.variables["u"][:], np.float32)[:, ::-1, :]
            v = np.asarray(ds.variables["v"][:], np.float32)[:, ::-1, :]
            if record["arm"] == "mapped":
                saved_n = np.asarray(ds.variables["manning"][:], np.float32)[::-1]
                max_n_error = max(max_n_error, float(np.max(np.abs(saved_n-n))))
            z_saved = np.asarray(ds.variables["zb"][:], np.float32)[::-1]
        if h.shape != (record["minutes"]//5+1, SIZE, SIZE):
            raise ValueError(f"Unexpected depth output in {record['name']}")
        if np.max(np.abs(z_saved-z)) > 1e-5:
            raise ValueError(f"Terrain mismatch in {record['name']}")
        for step in range(len(h)):
            expected = record["rate"]/1000*STEP_S/3600*step
            max_balance_mm = max(max_balance_mm, abs(float(h[step].mean()-expected))*1000)
        padded = np.zeros((5, SIZE, SIZE), np.float32)
        padded[:len(h)] = np.maximum(h, 0)
        padded_uv = np.zeros((5, 2, SIZE, SIZE), np.float32)
        padded_uv[:len(u), 0] = u
        padded_uv[:len(v), 1] = v
        zs.append(z)
        ns.append(n)
        hs.append(padded)
        uvs.append(padded_uv)
    if max_n_error > 1e-6 or max_balance_mm > .02:
        raise ValueError(f"Teacher audit failed: n={max_n_error}, balance={max_balance_mm}")
    np.savez_compressed(OUT / "sequences.npz", z=np.stack(zs), manning=np.stack(ns),
                        h=np.stack(hs), uv=np.stack(uvs), rate=np.asarray([x["rate"] for x in rows]),
                        length=np.asarray([x["minutes"]//5+1 for x in rows]),
                        region=np.asarray([x["region"] for x in rows]),
                        arm=np.asarray([x["arm"] for x in rows]),
                        name=np.asarray([x["name"] for x in rows]))
    audit = {"cases": len(rows), "regions": {r: sum(x["region"] == r for x in rows)
                                          for r in REGIONS},
             "max_manning_alignment_error": max_n_error,
             "max_volume_balance_error_mean_depth_mm": max_balance_mm}
    (OUT / "collection_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


def load_sequences():
    with np.load(OUT / "sequences.npz") as d:
        return {key: d[key] for key in d.files}


def train_expanded():
    data = load_sequences()
    rows = []
    for i in range(len(data["rate"])):
        if data["region"][i] != "bang_khae":
            continue
        for step in range(1, int(data["length"][i])):
            h = data["h"][i]
            x = features(data["z"][i], h[max(0, step-2)], h[step-1],
                         int(data["rate"][i]), data["manning"][i])
            rows.append((x, h[step][None].copy(), "train", int(data["rate"][i]),
                         str(data["arm"][i])))
    if not rows:
        raise ValueError("No Bang Khae training examples")
    train(extra_rows=rows, output_dir=OUT / "expanded_fit")


def evaluate_transfer():
    torch.set_num_threads(4)
    data = load_sequences()
    models = {}
    old_model = PlainSpatialCNN().eval()
    state = torch.load(OLD / "pretrained.pt", map_location="cpu",
                       weights_only=True)["states"]["plain_spatial_cnn"]
    old_model.load_state_dict(state)
    models["terrain_only"] = old_model
    for label, path in (("bang_kapi_surface", BASE / "pretrained.pt"),
                        ("bang_kapi_plus_bang_khae", OUT / "expanded_fit" / "pretrained.pt")):
        model = SurfaceSpatialCNN().eval()
        model.load_state_dict(torch.load(path, map_location="cpu",
                                         weights_only=True)["state"])
        models[label] = model
    paired_checkpoint = OUT / "paired_fit" / "pretrained.pt"
    if paired_checkpoint.exists():
        model = SurfaceSpatialCNN().eval()
        model.load_state_dict(torch.load(paired_checkpoint, map_location="cpu",
                                         weights_only=True)["state"])
        models["expanded_paired_rollout"] = model
    buckets = {}
    forecasts = {}
    for i in range(len(data["rate"])):
        if data["region"][i] != "lat_krabang":
            continue
        key = f"{data['arm'][i]}_{'unseen_60' if data['rate'][i] == 60 else 'seen_rates'}"
        target = data["h"][i, int(data["length"][i])-1]
        for label, model in models.items():
            previous = np.zeros((SIZE, SIZE), np.float32)
            current = previous.copy()
            for step in range(1, int(data["length"][i])):
                x = (_features(data["z"][i], previous, current,
                               int(data["rate"][i])) if label == "terrain_only" else
                     features(data["z"][i], previous, current, int(data["rate"][i]),
                              data["manning"][i]))
                with torch.no_grad():
                    following = model(torch.from_numpy(x[None]))[0, 0].numpy()
                previous, current = current, following
            wet = target > .02
            row = {"final_rollout_mae_mm": float(np.mean(np.abs(current-target))*1000),
                   "final_wet_gt20mm_mae_mm": float(np.mean(np.abs(current[wet]-target[wet]))*1000)
                   if wet.any() else None}
            buckets.setdefault(key, {}).setdefault(label, []).append(row)
            forecasts[(i, label)] = current
    summary = {}
    for key, arms in buckets.items():
        summary[key] = {}
        for label, rows in arms.items():
            summary[key][label] = {"cases": len(rows)}
            for metric in rows[0]:
                values = [r[metric] for r in rows if r[metric] is not None]
                summary[key][label][metric] = float(np.mean(values)) if values else None
    records = json.loads((OUT / "manifest.json").read_text())
    paired_gains = {}
    rng = np.random.default_rng(188)
    for group, arms in buckets.items():
        base_label, expanded_label = "bang_kapi_surface", "bang_kapi_plus_bang_khae"
        base_rows, expanded_rows = arms[base_label], arms[expanded_label]
        differences = np.asarray([a["final_rollout_mae_mm"]-b["final_rollout_mae_mm"]
                                  for a, b in zip(base_rows, expanded_rows)])
        matching = [r for r in records if r["region"] == "lat_krabang" and
                    f"{r['arm']}_{'unseen_60' if r['rate'] == 60 else 'seen_rates'}" == group]
        if len(matching) != len(differences):
            raise ValueError(f"Case alignment failed for {group}")
        tile_ids = [f"{r['row']}:{r['col']}" for r in matching]
        tile_means = np.asarray([np.mean(differences[np.asarray(tile_ids) == tile])
                                 for tile in sorted(set(tile_ids))])
        draws = rng.choice(tile_means, size=(3000, len(tile_means)), replace=True).mean(axis=1)
        paired_gains[group] = {
            "case_count": len(differences), "independent_tile_count": len(tile_means),
            "mean_gain_mm_positive_favors_extra_area": float(differences.mean()),
            "tile_mean_bootstrap_95_interval_mm": [float(x) for x in
                                                    np.quantile(draws, [.025, .975])],
            "cases_improved": int(np.count_nonzero(differences > 0)),
            "tiles_improved_on_average": int(np.count_nonzero(tile_means > 0))}
    index = {(r["region"], r["row"], r["col"], r["rate"], r["minutes"], r["arm"]): i
             for i, r in enumerate(records)}
    deltas = {}
    for key in sorted({(r["region"], r["row"], r["col"], r["rate"], r["minutes"])
                       for r in records if r["region"] == "lat_krabang"}):
        i0, i1 = index[(*key, "uniform")], index[(*key, "mapped")]
        group = "unseen_60" if key[3] == 60 else "seen_rates"
        truth = (data["h"][i1, int(data["length"][i1])-1] -
                 data["h"][i0, int(data["length"][i0])-1])*1000
        for label in models:
            predicted = (forecasts[(i1, label)]-forecasts[(i0, label)])*1000
            slot = deltas.setdefault(group, {}).setdefault(label, {"true": [], "pred": []})
            slot["true"].append(truth.ravel())
            slot["pred"].append(predicted.ravel())
    delta_summary = {}
    for group, arms in deltas.items():
        delta_summary[group] = {}
        for label, values in arms.items():
            truth = np.concatenate(values["true"])
            predicted = np.concatenate(values["pred"])
            delta_summary[group][label] = {
                "pairs": len(values["true"]),
                "teacher_mean_abs_surface_effect_mm": float(np.mean(np.abs(truth))),
                "predicted_mean_abs_surface_effect_mm": float(np.mean(np.abs(predicted))),
                "zero_effect_mae_mm": float(np.mean(np.abs(truth))),
                "predicted_effect_mae_mm": float(np.mean(np.abs(predicted-truth))),
                "effect_pearson": float(np.corrcoef(truth, predicted)[0, 1])
                if np.std(predicted) > 1e-10 else None}
    result = {"target": "synthetic SFINCS water depth, not observed flooding",
              "test_region": "Lat Krabang; untouched during weight fitting and selection",
              "same_architecture_initialisation_and_Bang_Kapi_validation": True,
              "added_training": "20 Bang Khae 960 m tiles, uniform and mapped roughness, 20/40/80 mm/h",
              "results": summary,
              "paired_extra_area_gains": paired_gains,
              "matched_surface_effect": delta_summary}
    (OUT / "transfer_evaluation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("static", "prepare", "collect",
                                           "train_expanded", "evaluate_transfer"))
    action = parser.parse_args().action
    {"static": static, "prepare": prepare, "collect": collect,
     "train_expanded": train_expanded,
     "evaluate_transfer": evaluate_transfer}[action]()
