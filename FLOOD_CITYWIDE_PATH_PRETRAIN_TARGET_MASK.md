# Path pretraining followed by target-masked report training

**Tested 23 October 2026.** The earlier matched-path pretraining experiment
fine-tuned on genuinely isolated report sites, but did **not** use the later
target-history masking augmentation. This experiment answers whether combining
them helps the strict emerging-site queue.

## Matched experiment

The saved path model was pretrained in each geographic fold to rank a synthetic
history source connected to a target by 2–6 LiDAR D8 hops above an equally
distant, elevation-matched non-path source. Its synthetic source channel was
mapped to the 365-day report-history channel. The five spatial steps and
LiDAR/static input weights were transferred; the other raw-history and five
wide-history weights began at zero, and the report head was reset.

For each fold and fine-tuning seed, a **scratch control** received the same
LiDAR, canal, road, cover, and report-history layers; same 2023–2024 source
district report examples; same optimizer, two epochs, group-level ranking loss,
and source-only selection of the CNN correction strength. Both arms included
lower-weight first-in-a-year adjacent-history groups. Every positive and
unreported comparison patch had its target-centered 3 × 3 raw report history
removed, including the exact contribution of those cells to all five Gaussian
history layers. The target-mask recomputation and isolated inference checks
passed. The three runs reuse the **same fold-specific synthetic pretraining
checkpoints** and vary only the report fine-tuning seed. At inference, strict
isolated cells already have zero history in that 3 × 3 neighborhood.

Each Q4 2025 held-out district fold uses a daily 10% cell inspection budget,
with 20% of slots reserved for cells having no report in their own or eight
adjacent cells during the prior 365 days. There are **332** strict isolated
report-positive cell-days and **158** touching positive groups across folds.
The source-selected correction is the primary comparison; no Q4 result enters
that selection.

| Fine-tuning seed | Distant history cells / groups | Scratch masked CNN cells / groups | Path-pretrained masked CNN cells / groups |
| --- | ---: | ---: | ---: |
| 0 | 65 / 44 | 53 / 41 | 62.78 / 41 |
| 10,000 | 65 / 44 | 63 / 44 | **68.78 / 46.78** |
| 20,000 | 65 / 44 | 64 / 49 | 60.34 / 45.34 |
| Mean across these runs | 65 / 44 | 60 / 44.67 | 63.97 / 44.37 |

The fractional hits come from fractional tie allocation in the fixed queue.
The mean is a **stability summary over repeated fits on the same Q4 data**,
not three independent test sets. Path pretraining improves cell hits over
matched scratch in two seeds and worsens them in one. Its mean **63.97/332
(19.27%)** is below simple distant history's **65/332 (19.58%)**. Group capture
does not improve over scratch on average. With Q4 dates resampled in paired
seven-day circular blocks, the mean pretraining-minus-scratch cell-capture
difference is **+1.19 percentage points**, with an exploratory 95% interval of
**−2.83 to +4.56 points** (10,000 draws, random seed 714, the same dates
resampled for all folds and seeds). This interval also reflects the limited Q4
positive sample and should not be treated as a fresh holdout guarantee.

For context, keeping the CNN correction fixed at `0.1` after viewing Q4 would
give the seed-0 pretrained model **68/332 cells and 46/158 groups**. The
source-validation-selected correction for that same seed gives **62.78/332
and 41/158**. That Q4-selected diagnostic is not the operational result.

## Reading

The combination can sometimes use the spatial initialization better than
target masking alone, but it has **not** shown a stable gain over distant
report history for new report sites. The earlier unmasked path transfer found
only **42.78/332** with its report head reset, although its architecture and
report loss differ from the masked group model here, so those numbers are not a
matched estimate of the masking effect. The matched estimate is the
pretrained-versus-scratch comparison within each row above.

The target is a **future Traffy report-positive cell**, not observed wetness.
Unreported comparisons are unknown rather than verified dry. D8 routes are
terrain proxies, not checked drains or water flow. Present-day LiDAR and
static features are not vintage-matched. Q4 2025 has been inspected repeatedly
during this project, so these findings are exploratory. They do not validate
flood probabilities or rainfall counterfactuals.

A later [joint target-type experiment](FLOOD_CITYWIDE_JOINT_PATH_MASK.md)
shared this masked ranker's spatial steps with a pre-Q4 UNOSAT mapped-water
head. It produced one promising seed but no repeatable Bangkok gain; a
satellite spatial-label shuffle control also changed sign across seeds.

An [ensemble of the three saved path-masked CNNs](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_path_mask_three_seed_ensemble_fivefold.json)
was also tested without retraining. Each fold chose the mean residual's
correction strength from January–September 2025 source districts. All five
chose `0.25`; the held-out Q4 queue found **60.34/332 isolated cells and
42.34/158 touching groups**, below distant history's **65/332 and 44/158**.
A fixed `0.1` diagnostic found **64.84/332 and 45.78/158**, but that weaker
setting was not selected by source validation and is not a new operational
win. Averaging seeds alone does not repair the validation-to-Q4 shift.

## Reproduce

- [Matched training and evaluation code](flood_citywide_path_pretrain_mask_transfer.py)
  reuses [the exact target-mask/group trainer](flood_citywide_isolated_group_mask.py)
  and the saved synthetic states from the [path experiment](FLOOD_CITYWIDE_PATH_CONTRAST_PRETRAIN.md).
- Run `.venv/bin/python flood_citywide_path_pretrain_mask_transfer.py`, then
  repeat with `--seed-offset 10000` and `--seed-offset 20000`.
- [Seed 0](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_path_pretrain_target_mask_matched_fivefold.json),
  [seed 10,000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_path_pretrain_target_mask_matched_fivefold_seed_10000.json),
  and [seed 20,000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_path_pretrain_target_mask_matched_fivefold_seed_20000.json)
  contain per-fold source validation, held-out daily hits, and selected
  correction strengths; matching `.pt` files hold model weights.
