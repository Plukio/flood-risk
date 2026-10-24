# BMA road-corridor pretraining for emerging report sites

**Exploratory backtest, 23 October 2026.** This test asks whether real BMA
road-flood reports teach the spatial CNN more useful structure than its earlier
synthetic terrain-path pretraining alone. The outcome remains a **future Traffy
report-positive cell**, not verified water extent or flood probability.

## Road labels and model

The BMA 2022–2024 CSVs contain **772 depth-bearing records**. One has a district
outside the modeled district catalog, leaving **771** eligible rows. Named OSM
roads in the same district match 658 of the original 772; one touches no 250 m
candidate cell, leaving **657 rasterized records**. Each matched road is a bag
of every candidate cell its named OSM geometry touches within the district.
It is an **uncertain corridor**: the report does not say which member cell was
wet. The free-text area field was not geocoded.

The first fit keeps roads spanning **2–30 candidate cells**: 570 records across
96 unique reported district-road bags. A deterministic quarter-sample of
other named OSM roads provides same-district, similar-size comparison bags.
Those roads have no matched BMA record in these files, but they are **unknown,
not verified dry**. Comparisons crossing more than a quarter of the shorter
road's cells are discarded. Each geographic source fold has **63–79 usable
reported-versus-comparison road pairs**, representing 28–30 districts.

The model begins with the saved fold-specific synthetic-path CNN. For three
epochs, it sees source-district road bags with **LiDAR terrain, mapped roads and
canals, 2021 WorldCover cover, grid validity and candidate geometry**. All
report-history inputs are zero during this stage. It learns to put the
reported road's normalized `0.05 × logmeanexp(cell score / 0.05)` above its
comparison road. Each unique reported road is used once per epoch, so a road
with many BMA records does not dominate the updates. The normalized bag score
avoids rewarding a road merely for containing more cells.

The pretrained spatial weights then enter the **same target-masked Traffy
trainer** as the matched path-only control. That trainer uses 2023–2024
source-district reports, removes the scored cell's 3×3 prior report history
from raw and Gaussian-smoothed channels, and trains on reported groups versus
unreported comparisons. Correction strength is selected from January–September
2025 source districts; Q4 2025 districts remain held out from weight fitting
and selection. Each day's queue covers 10% of candidate cells, with 20% of
slots reserved for strict emerging candidates. The BMA labels are restricted
to each fold's source districts. No Q4 result is used to pick a seed or weight.

## Held-out Q4 results

The table reports fractional queue hits among **332** strict isolated
report-positive cell-days and **158** touching positive groups in the same
five geographic folds. Fractional hits arise from boundary ties. The three
seeds repeat fits on the **same Q4 dates**; they are not separate test sets.

| Fine-tuning seed | Distant history cells / groups | Path-only masked CNN | Real BMA road bags | Shuffled road labels |
| --- | ---: | ---: | ---: | ---: |
| 0 | 65 / 44 | 62.78 / 41 | **70 / 44** | 63 / 41 |
| 10,000 | 65 / 44 | 68.78 / 46.78 | **71.78 / 51.78** | 59.34 / 41 |
| 20,000 | 65 / 44 | 60.34 / 45.34 | **64.78 / 47.78** | 49 / 35 |
| Mean over fits on the same Q4 | 65 / 44 | 63.97 / 44.37 | **68.85 / 47.85** | 57.11 / 39 |

The shuffled control randomly flips which road is called reported in half of
the same matched pairs, fixed across epochs. It preserves the number of pairs
and road sizes but removes the expected relationship between the BMA label and
spatial inputs. Real labels beat this control in all three runs. Real BMA
pretraining also beats the matched path-only CNN on isolated cell hits in all
three runs, by **4.88 hits on average**. It beats distant history in two of
three runs and ties or beats it on group hits in all three.

The mean real-minus-path **cell-capture** difference is **+1.47 percentage
points**. A paired Q4 seven-day block resampling interval is **−0.62 to +2.63
points**. Real-minus-history is **+1.16 points**, with interval **−4.40 to
+5.18 points**. Both include zero. Real-minus-shuffled is **+3.54 points**,
with interval **+1.94 to +5.04 points**. These exploratory intervals resample
the same 92 Q4 days across folds and seeds; they cannot account for the many
design decisions already informed by this quarter.

**Reading:** road-level weak supervision is a promising transfer signal for
the masked CNN, and its labels matter relative to the shuffled check. It is
not yet a reliable improvement over the simple distant-history queue. In one
seed the real-label model finds **64.78/332** isolated cells, slightly below
history's **65/332**. The test also does not show that any road-bag cell was
actually wet, that the model knows water flow, or that it can simulate more
rain. Another untouched season and better BMA corridor localization would be
needed to establish an operational gain.

## Reproduce and limits

- Build the [road-bag cache](flood_citywide_bma_road_bags.py) with
  `.venv/bin/python flood_citywide_bma_road_bags.py`.
- Run [matched pretraining and fine-tuning](flood_citywide_bma_road_pretrain.py)
  three times with seed offsets `0`, `10000`, and `20000`.
- Generate the [three-seed summary](flood_citywide_bma_road_summary.py) with
  `.venv/bin/python flood_citywide_bma_road_summary.py`.
- Inspect the [road audit](artifacts/flood_open_weak_train/multi_target/citywide/bma_road_bags_2022_2024_audit.json),
  [per-fold seed-0 record](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_bma_road_pretrain_mask_fivefold.json), and
  [summary](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_bma_road_pretrain_mask_three_seed_summary.json).

OSM geometry is from 2025 rather than each event date, BMA street names can
be ambiguous, and the road-level label includes neither verified dry cells
nor a precise wet segment. Static cover can also encode where roads and people
are observed. The simple history reference is still the default queue until
a fresh holdout confirms a gain.
