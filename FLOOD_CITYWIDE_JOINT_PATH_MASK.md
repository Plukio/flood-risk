# Joint mapped-water and target-masked report training

**Tested 23 October 2026.** This experiment asks whether an additional
mapped-water target teaches the path-pretrained CNN spatial context that
transfers to **new Bangkok report sites**. The main result is **no stable
transfer gain**. One fine-tuning seed improved, two did not, and an equal-step
spatial-label shuffle control showed no consistent benefit from the satellite
map's true spatial alignment.

## Matched design

The four Bangkok arms cross two choices:

| Initialization | Report-only training | Joint training |
| --- | --- | --- |
| Scratch | Exact 3 × 3 target-masked report group loss | Same loss, interleaved with mapped-water auxiliary updates |
| Fold-specific matched-path pretraining | Same report loss | Same report loss and auxiliary updates |

All arms use the **same fold-specific LiDAR, roads, canals, cover, four pre-day
report histories, and five wide report-history layers**. The report trainer
removes the target's 3 × 3 local history and its exact contributions to the
wide Gaussian layers for both report positives and unreported comparisons.
This keeps the training input consistent with strict isolated inference.
Source-district January–September 2025 selects the CNN correction strength;
Q4 2025 outcomes in excluded districts are used only for scoring.

The joint model has a separate satellite static-input adapter and UNOSAT
output head. Its **five spatial convolutional steps are shared** with the
Bangkok report branch. One UNOSAT update follows every two report updates.
The satellite scenes use source-region normalization, while the report branch
keeps its source-district LiDAR normalization. The auxiliary target is
source-specific mapped wet area plus UNOSAT's analyzed non-water evidence;
unreported Bangkok cells are never treated as observed dry.

To retain a clean pre-Q4 training cutoff, the auxiliary set contains only
**54 UNOSAT 2021 scenes and one 2024 scene across 17 non-Bangkok regions**.
Bangkok scenes and all **four Pathum Thani 2021 scenes** are withheld from
auxiliary fitting; 2025 November EOS and all 2026 GISTDA/EOS labels are also
excluded. The Pathum Thani holdout shares storm dates with other provinces,
so it tests held-out geography rather than a new storm. Only positive mapped
area is counted in its top-10% ranking check.

## Bangkok held-out Q4 result

The fixed daily queue selects 10% of 250 m cells within each held-out district
fold and reserves 20% of slots for cells with **no own or adjacent report in
the previous 365 days**. Across five folds it contains **332** isolated
report-positive cell-days and **158** touching positive groups. These are
report outcomes, not verified flooded or dry cells. Correction strength was
chosen on source-district validation for every model.

| Fine-tuning seed | Scratch report-only cells / groups | Path report-only cells / groups | Scratch joint cells / groups | Path joint cells / groups | Path joint with spatially shuffled UNOSAT labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 53 / 41 | 62.78 / 41 | 58 / 38 | **72.8 / 47.46** | 65 / 43 |
| 10,000 | 63 / 44 | **68.78 / 46.78** | 70 / 47 | 56.78 / 42.78 | 69 / 49 |
| 20,000 | 64 / 49 | 60.34 / 45.34 | 62 / 47 | 58 / 44 | 52.78 / 40.78 |
| Three-fit mean | 60 / 44.67 | 63.97 / 44.37 | 63.33 / 44 | 62.53 / 44.75 | 62.26 / 44.26 |

The **simple distant-history score finds 65/332 cells and 44/158 groups** on
every one of these identical queues. The three-fit mean for path joint is
**62.53/332 (18.83%)**, below distant history's **19.58%**. Joint training
changes the path model by **+10.02, −12, and −2.34 cells** relative to path
report-only across seeds. Its group advantage over history is similarly
small and inconsistent.

The shuffle control retains every satellite scene's wet and analyzed-area
mass, all additional optimizer updates, the same model, and the same scene
draws, but permutes mapped labels across its cells before training. Real
minus shuffled satellite labels changes Bangkok cell hits by **+7.8, −12.22,
and +5.22** across the three seeds. The mean is **+0.27 cells**, or **+0.08
percentage points**. A paired seven-day circular-block resampling of Q4 dates
gives an exploratory 95% interval of **−2.07 to +2.19 points** for that
three-fit mean difference (10,000 draws, random seed 714). These seeds reuse
the same Q4 outcome set; they are stability checks, not independent tests.

## Did the satellite task learn anything?

Yes, on its own target. On the **withheld Pathum Thani UNOSAT scenes**, the
real-label path-joint satellite head captured **66.72%, 65.76%, and 67.12%**
of mapped positive area within its top 10% of cells, averaged over the five
fold-specific models. The simple crop-cover rank captured **67.08%**.
Shuffled-label models fell to **5.06%, 3.58%, and 10.50%**. This shows that
real satellite labels gave the auxiliary head usable spatial alignment, but
the resulting head did **not** clearly outperform crop cover, and the shared
spatial steps did **not** show consistent transfer to Bangkok reports.

The satellite mapped-water target is often broad agricultural water, while
the Bangkok target is a later **report-positive street cell**. Separate input
adapters prevent LiDAR/SRTM channel mismatch; they do not make the label
meanings equivalent. Only UNOSAT pre-Q4 scenes were available for this clean
time cutoff, so this is a two-target-type experiment, rather than a full
UNOSAT/GISTDA/EOS/BMA joint run. Including post-Q4 imagery in a Q4 backtest
would weaken a forward-time claim. Present-day LiDAR and other static layers
are not vintage-matched, D8 paths are terrain proxies rather than validated
drains, and Q4 has been inspected repeatedly during this project.

**Decision:** This joint recipe is useful evidence about the transfer gap,
but it does not beat distant history reliably or validate a flood-probability
or rainfall-scenario model. More varied, time-matched **urban observed wet and
dry areas** would be a closer auxiliary target than the current mapped-water
scenes.

## Reproduce and inspect

- [Joint training, source holdouts, and label-shuffle control](flood_citywide_joint_path_mask.py).
- Run `.venv/bin/python flood_citywide_joint_path_mask.py`; repeat with
  `--seed-offset 10000 --reuse-report-controls` and
  `--seed-offset 20000 --reuse-report-controls`. The reused report-only arms
  come from the [earlier matched path-mask experiment](FLOOD_CITYWIDE_PATH_PRETRAIN_TARGET_MASK.md);
  seed-0 report-only arms were retrained here and reproduced exactly.
- Run the shuffle control with
  `.venv/bin/python flood_citywide_joint_path_mask.py --shuffle-satellite-labels --path-joint-only`
  and repeat with each `--seed-offset`.
- Joint result artifacts: [seed 0](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_joint_unosat_path_mask_fivefold.json),
  [seed 10,000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_joint_unosat_path_mask_fivefold_seed_10000.json),
  [seed 20,000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_joint_unosat_path_mask_fivefold_seed_20000.json).
  Matched [shuffled seed 0](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_joint_unosat_path_mask_fivefold_shuffled_satellite_labels.json),
  [seed 10,000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_joint_unosat_path_mask_fivefold_seed_10000_shuffled_satellite_labels.json),
  and [seed 20,000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_joint_unosat_path_mask_fivefold_seed_20000_shuffled_satellite_labels.json)
  hold the controls. Matching `.pt` files save model states.
