# SFINCS spatial pretraining with flow direction and masking

**Exploratory simulator experiment, 23 October 2026.** The strongest corrected
model is a 10 m dilated CNN that imitates five-minute SFINCS water movement.
We tested whether velocity-direction labels and random gaps in the older depth
map help it transfer to new tiles and an unseen, longer storm. The targets are
**simulated depth and velocity**, not measured flooding.

## What entered the model

Each 960 m × 960 m tile has 96 × 96 cells. The input layers are relative LiDAR
elevation, local relief, depth at the previous and current five-minute states,
rain rate, and assumed Manning roughness from WorldCover and OSM road cover. A
seventh layer marks missing pixels in the older depth state. The CNN predicts
the next depth and east/north velocity components. It keeps predicted depth
nonnegative and rescales the whole tile to equal current water plus rain.

The SFINCS v2.4.1 cases start dry with uniform rain and closed tile edges.
They have no pipe, pump, canal-level, infiltration, or building-barrier
response. WorldCover/road roughness values are scenario assumptions, not
measured local coefficients. Corrected [Bang Kapi collection audit](artifacts/flood_bangkapi_surface/sfincs_surface_pretrain/collection_audit.json)
and [Bang Khae/Lat Krabang collection audit](artifacts/flood_bangkapi_surface/sfincs_surface_multiarea/collection_audit.json)
both show **zero Manning-grid misalignment** and less than 0.000005 mm mean
depth water-balance discrepancy. Surface checkpoints trained before this
alignment fix were retrained; the refreshed [Bang Kapi evaluation](artifacts/flood_bangkapi_surface/sfincs_surface_pretrain/evaluation.json)
and [cross-area surface evaluation](artifacts/flood_bangkapi_surface/sfincs_surface_multiarea/transfer_evaluation.json)
use these corrected outputs.

Bang Kapi and Bang Khae supply **222 paired training cases**. A separate
Bang Kapi geographic slice supplies **27 validation pairs**. The tests use
18 other Bang Kapi tiles and **all 20 Lat Krabang tiles**, with both uniform
and mapped roughness. Training and checkpoint selection use 20, 40 and
80 mm/h rain for 15 minutes. **60 mm/h for 20 minutes** is reserved for the
geographic tests; its validation-tile copy was excluded from fitting.

## Training comparison

All four rows below start with the same corrected paired rollout checkpoint.
The source row is unchanged. Each other row receives 12 more epochs, the
same training cases and sample order, and the original rollout loss:
depth-weighted absolute error plus twice the error in the mapped-minus-uniform
depth difference. Checkpoints are selected using that **unmasked** Bang Kapi
validation objective.

| Arm | Added training |
| --- | --- |
| Corrected paired source | None |
| Matched depth-only | Continue paired depth and surface-difference loss |
| Depth + direction | Also predict SFINCS velocity; normalized vector Huber and cosine direction loss act where target speed >0.01 m/s and depth >5 mm |
| Depth + direction + mask | Same losses; at the third rollout step, hide one or two 80–240 m blocks in the **older** depth layer on 45% of training pairs |

The same mask covers the uniform and mapped members of a pair. Current depth
stays visible because the output's whole-tile water budget depends on it.
Thus the mask test probes missing *older state* and cannot prove the model can
reconstruct missing current water. The [validation curves](artifacts/flood_bangkapi_surface/sfincs_direction_pretrain/paired_extension/paired_training_curves.png)
and [training records](artifacts/flood_bangkapi_surface/sfincs_direction_pretrain/paired_extension)
show that the matched depth-only arm has the lowest paired validation loss
(5.835 mm); direction-only and direction-plus-mask reach 5.869 and 5.870 mm.

## Held-out full-storm results

Each number is **final water-depth mean absolute error in mm over all 10 m
cells**, averaged equally across complete cases. The model feeds its own
predicted depths into the next five-minute step. Smaller is better.

| Model | Bang Kapi, trained 15-min storms | Bang Kapi, unseen 20-min storm | Lat Krabang, trained 15-min storms | Lat Krabang, unseen 20-min storm |
| --- | ---: | ---: | ---: | ---: |
| Corrected paired source | 2.359 | 5.115 | 2.320 | 5.008 |
| **Matched depth-only** | **2.299** | 4.933 | **2.259** | 4.826 |
| Depth + direction | 2.313 | 4.981 | 2.278 | 4.897 |
| Depth + direction + mask | 2.308 | **4.861** | 2.271 | **4.758** |

Direction plus masking improves the unseen Lat Krabang storm by **0.068 mm
(1.4%)** against matched depth-only, but is **0.012 mm (0.5%)** worse on
its trained storms. Direction supervision alone loses to matched depth-only
on all four full-storm groups. Masking helps most when paired with the longer,
unseen rain scenario. Full numbers, including cells above 20 mm depth, are in
the [rollout evaluation](artifacts/flood_bangkapi_surface/sfincs_direction_pretrain/paired_extension/rollout_evaluation.json).
On the unseen Lat Krabang storm, the mean error **within simulated cells
deeper than 20 mm** also falls, from 12.370 to 12.267 mm.

The [per-tile audit](artifacts/flood_bangkapi_surface/sfincs_direction_pretrain/paired_extension/tile_bootstrap.json)
shows that direction plus masking beats matched depth-only on **20/20 Lat
Krabang** tiles and **17/18 Bang Kapi** tiles for the unseen storm. The mean
tile differences are −0.069 and −0.072 mm respectively; 20,000 tile
bootstrap resamples give intervals of [−0.081, −0.056] and [−0.088, −0.054]
mm. Neighboring tiles can be correlated, so those intervals only describe
this synthetic test. On trained storms, masking loses on 18/20 Lat Krabang
and 16/18 Bang Kapi tiles by about 0.01 mm.

For the unseen Lat Krabang rollout, the direction-only head has **33.7°**
mean angular error on moving wet cells; direction plus mask has **32.9°**.
Their fractions within 30° are **63.5%** and **64.4%**. The held-out
[one-step evaluation](artifacts/flood_bangkapi_surface/sfincs_direction_pretrain/paired_extension/evaluation.json)
also tests deliberately hidden older-depth blocks: on the unseen Lat Krabang
storm, MAE is 1.452 mm with the masked-trained model versus 1.543 mm with
matched depth-only. This measures robustness to missing prior state, not
verified physical flow.

An earlier 24-epoch next-step comparison started from the uniform-terrain
checkpoint and also found a longer-storm gain from masking. Its valid but
weaker [full-storm results](artifacts/flood_bangkapi_surface/sfincs_direction_pretrain/rollout_evaluation.json)
are kept separately; the corrected paired comparison above is the stronger
test of added value.

## Interpretation and next use

The model can learn a simulator velocity direction signal. Randomly hiding
the older water state makes its longer-storm prediction more stable across
the held-out area. The depth gain is consistent across these synthetic tiles
but small in millimetres, and trained-storm depth loses slightly. This does
not establish accuracy on Bangkok flood extent, depth, reporting, or a real
storm. Tile-edge pooling and missing drainage can dominate those outcomes.

Keep the corrected paired depth-only continuation as the everyday simulator
reference, and carry the direction-plus-mask arm as the candidate for
longer-storm and missing-state robustness. A future field-value test needs
time-matched observed wet **and dry** cells or depth measurements before
either model can be called a flood probability or operational forecast.

## Reproduce

The corrected SFINCS cases are already collected in the artifact directories.
The surface preparation and collection code is in
[flood_sfincs_surface_pretrain.py](flood_sfincs_surface_pretrain.py) and
[flood_sfincs_surface_multiarea.py](flood_sfincs_surface_multiarea.py).
The training and evaluation code is in
[flood_sfincs_surface_pair_train.py](flood_sfincs_surface_pair_train.py),
[flood_sfincs_direction_pretrain.py](flood_sfincs_direction_pretrain.py), and
[flood_sfincs_direction_paired_extension.py](flood_sfincs_direction_paired_extension.py).

```sh
.venv/bin/python flood_sfincs_surface_multiarea.py train_expanded
.venv/bin/python flood_sfincs_surface_pair_train.py
.venv/bin/python flood_sfincs_direction_pretrain.py
.venv/bin/python flood_sfincs_direction_paired_extension.py
```
