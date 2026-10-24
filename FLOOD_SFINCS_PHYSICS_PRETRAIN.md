# SFINCS physics pretraining on Bang Kapi LiDAR

**Exploratory synthetic experiment, 23 October 2026.** We trained spatial
CNNs to imitate 5-minute water-depth transitions from the official SFINCS
v2.4.1 solver. This is a physics **pretraining stage** for a future flood
model. It has not been fine-tuned on Bangkok reports or verified flood extent.

## Target and input

The [Bang Kapi 10 m DTM](data/flood_bangkapi_surface/dtm_bangkapi_10m.tif)
was divided into **81 nonoverlapping 96 × 96 cell tiles**. SFINCS ran 270
tile–storm cases and wrote depth at 0, 5, 10, 15 and, for the reserved storm,
20 minutes. This yielded **837** next-state examples. Its setups use uniform
rain, initially dry cells, closed tile boundaries, Manning n = 0.035, no
infiltration, and no drains or canal boundary levels. The greatest water
balance discrepancy over all saved 5-minute states was **0.0000047 mm** of
mean cell depth.

At each step, the network receives the 10 m relative elevation, local relief
(elevation minus a Gaussian neighborhood mean), previous depth, current
depth, and rain rate. It predicts the **next 10 m depth map**. The output uses
a nonnegative activation and is normalized so its sum equals current water
plus the 5-minute rain input. This enforces whole-tile storage for the closed
domain. The weighted cell L1 loss gives extra weight to deeper cells. No
report label, satellite water label, or measured flood depth entered training.

We compared a three-level **U-Net** with a **plain full-resolution spatial CNN**
using dilated convolutions. A 1 × 1 convolutional control gets the **same
input and water-volume rule**, including the handcrafted local relief, but
cannot learn additional spatial context. Uniform rain accumulation without
flow is another control. Source tiles are columns 0–5 (54 tiles),
geographic validation is column 6 (9 tiles), and final test is columns 7–8
(18 tiles). Training storms are **20, 40 and 80 mm/h for 15 minutes**. The
**60 mm/h for 20 minutes** storm is absent from weight fitting and epoch
selection. Epochs are selected by one-step MAE on geographic validation tiles
at the training rates. The saved U-Net, plain CNN and local control epochs are
79, 79 and 80. Augmentation rotates and reflects paired input and target maps.

## Held-out simulator agreement

All errors below are millimetres of depth, averaged equally over cases.
“Final rollout” means the model feeds its own predicted states back through
all three or four 5-minute steps. The deeper-cell column evaluates SFINCS
cells above 20 mm at the last step.

| Held-out test | Model | Final rollout MAE, all cells | Final rollout MAE, central 64 × 64 | Final MAE where SFINCS depth >20 mm | One-step MAE with true preceding states |
| --- | --- | ---: | ---: | ---: | ---: |
| 18 unseen tiles, three trained rain rates; 54 cases | **Plain spatial CNN** | **3.45** | **3.35** | **13.21** | **0.91** |
| Same | U-Net | 3.60 | 3.44 | 14.09 | 1.01 |
| Same | Local 1 × 1 | 8.94 | 8.99 | 26.57 | 1.79 |
| Same | Uniform accumulation | 12.70 | 12.69 | 36.24 | — |
| 18 unseen tiles, unseen 60 mm/h × 20 min; 18 cases | **Plain spatial CNN** | **6.86** | **6.70** | **20.21** | **1.52** |
| Same | U-Net | 7.49 | 7.18 | 22.60 | 1.67 |
| Same | Local 1 × 1 | 15.84 | 15.94 | 35.79 | 2.83 |
| Same | Uniform accumulation | 23.29 | 23.28 | 48.65 | — |

The plain spatial CNN reduces final simulator-imitation error by about **61%**
versus the trained local control for seen rain rates, and **57%** for the
unseen storm. It is also slightly better than the U-Net in both held-out tests,
so **U-Net complexity is not justified by this experiment**. All learned
arms have an autoregressive whole-tile storage error below **0.000006 mm**
of mean depth. The [validation curve](artifacts/flood_bangkapi_surface/sfincs_physics_pretrain/validation_curve.png)
and a [fixed held-out example](artifacts/flood_bangkapi_surface/sfincs_physics_pretrain/heldout_example.png)
show the training and map comparison. The [visual training guide](FLOOD_SFINCS_PRETRAIN_VISUAL_GUIDE.md)
shows the actual input layers, architecture, target and loss. Full numbers are in the
[evaluation artifact](artifacts/flood_bangkapi_surface/sfincs_physics_pretrain/evaluation.json),
with [weights](artifacts/flood_bangkapi_surface/sfincs_physics_pretrain/pretrained.pt),
[training curve data](artifacts/flood_bangkapi_surface/sfincs_physics_pretrain/training.json),
SFINCS input files, outputs and run logs beside them.

## What this establishes

The spatial models learned a transferable response **within the withheld part
of this one LiDAR footprint** and reproduce SFINCS better than a model without
learned spatial context. The result is about imitating the simulator. It does
not measure real flood prediction, downstream report ranking, or transfer
outside Bang Kapi. No cross-city data were used in this stage.

Whole-tile conservation does not constrain individual face flows: either spatial CNN
can mathematically move water farther than a physical flow path allows in
one step. Closed edges and 960 m tiles can also create artificial pooling.
SFINCS itself has no verified BMA drain connections, pump state, or event-time
canal levels in these cases. A future physical teacher should use buffered
domains, local flux checks, wider terrain coverage and observed water data.

The current masked **250 m Bangkok report CNN** has different input channels,
spatial scale and architecture. Copying either 10 m CNN's weights into it
would change the meaning of a convolution. A valid downstream test needs a
shared high-resolution encoder with an explicit 10 m → 250 m aggregation, or
a physics teacher generated at the report grid scale, followed by matched
report fine-tuning and a held-out Bangkok queue comparison. Until that test,
the pretraining result is **simulator skill only**.

## Reproduce

The setup, collection, training and evaluation are in
[flood_sfincs_physics_pretrain.py](flood_sfincs_physics_pretrain.py); the two
spatial network definitions and local control are in
[flood_sfincs_physics_model.py](flood_sfincs_physics_model.py).
The Python environment needs `torch`, `rasterio`, `scipy`, `matplotlib`, and
`netCDF4`; the SFINCS executable runs in the pinned Docker image.

```sh
.venv/bin/python flood_sfincs_physics_pretrain.py prepare
colima start --vm-type vz --vz-rosetta --cpus 4 --memory 6 --disk 30
docker run --rm --platform linux/amd64 -v "$PWD/artifacts/flood_bangkapi_surface/sfincs_physics_pretrain:/data" -w /data deltares/sfincs-cpu:sfincs-v2.4.1 /bin/sh /data/run_all.sh
.venv/bin/python flood_sfincs_physics_pretrain.py collect
.venv/bin/python flood_sfincs_physics_pretrain.py train
.venv/bin/python flood_sfincs_physics_pretrain.py evaluate
.venv/bin/python flood_sfincs_physics_pretrain.py plot
.venv/bin/python flood_sfincs_physics_pretrain.py plot_io
```
