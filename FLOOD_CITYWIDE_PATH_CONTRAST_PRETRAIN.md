# Matched-path spatial pretraining for emerging report sites

**Run 23 October 2026.** The earlier route pretraining taught a local D8
direction or total arriving runoff, but did not improve geographically held-out
flood-report ranking. This test forces a 15 × 15-patch CNN to distinguish a
**multi-cell connected path** from a matched non-path before fine-tuning it on
the same strict new-site report task.

## Pretraining task

For a target 250 m cell, construct two patches with the **same target terrain,
roads, canals, cover, and common distractor report-like points**:

| Patch | Synthetic source |
| --- | --- |
| Connected | A source 2–6 D8 downhill hops upstream of the target. |
| Comparison | A source at exactly the same squared grid distance from the target, within **0.5 m** of the connected source's elevation, whose D8 path does not reach the target within 20 hops. |

Both sources are outside the target's 3 × 3 neighborhood. The source occupies
the channel later used for 365-day report history. The pretraining loss is
`mean(softplus(score_comparison − score_connected))` plus a small
`0.0005 × mean(score²)` penalty. No reported flood site or satellite water
label enters this pretraining target. The path label is generated from cached
BMA LiDAR averaged to 250 m; it **does not represent verified drainage**.

There are **11,134** matched source-target pairs on the Bangkok grid. The
connected and comparison source elevations differ by a median **0.079 m**
(90th percentile **0.300 m**). Each geographic fold trains for 600 batches on
source-district pairs and tests the synthetic task on pairs wholly inside
excluded districts.

| Synthetic pair check | Result across excluded-district folds |
| --- | ---: |
| CNN ranks connected source above comparison | **54.4–67.1%**; **62.4%** pooled across 7,970 checked pairs |
| Same trained CNN with its three LiDAR channels set to source-normalized zero | **49.8–56.1%**; **53.5%** pooled |
| Equal-distance blind choice | 50% |

The difference after LiDAR removal is evidence that the CNN uses terrain
context for this **synthetic pair task**. The 54.4% fold shows that even this
pretraining skill is not stable across excluded areas.

## Transfer to the strict new-site queue

The source-channel weight becomes the 365-day report-history input weight;
the other three report-history input weights start at zero. Fine-tuning uses
the same 2023–2024 source-district isolated report examples and two-epoch
pairwise ranking loss as the previous LiDAR CNN. One arm keeps the synthetic
output head. A matched arm resets only that head, because the full-transfer
report loss started unusually high. Neither arm was selected using Q4.

The daily Q4 2025 evaluation selects 10% of held-out-fold candidate cells,
reserving 20% of slots for sites with no report in their own or eight adjacent
cells during the prior year.

| Ranker | Isolated positive cells found / 332 | Touching positive groups found / 158 |
| --- | ---: | ---: |
| **Distant report history** | **65 (19.58%)** | **44 (27.85%)** |
| LiDAR CNN trained from scratch | 48 (14.46%) | 30 (18.99%) |
| Matched-path pretrain, keep head | 40 (12.05%) | 31 (19.62%) |
| Matched-path pretrain, reset head | 42.78 (12.89%) | 27 (17.09%) |

The fractional 42.78 is the same queue's fractional tie allocation. Resetting
the head lowers the starting report-training loss and slightly improves cell
capture over full transfer, but neither transfer beats scratch or distant
history. On January–September 2025 **source districts**, reset-head capture
was 472.46/3,508 (13.47%), slightly above distant history's 463.18 (13.20%).
That source-district gain did not transfer to excluded Q4 districts.

## Why this relation has little reach in the report task

Across the **332** strict isolated Q4 positive cell-days, only **13** had a
prior-year report on a 2–6-hop D8 upstream path. In contrast, **323** had some
prior report within a broad six-cell square around the target. The broad
square is a coverage diagnostic, not the exact Gaussian nearby-history score.
Among all 965,680 eligible isolated cell-days, 29,866 had path-linked report
history. The explicit source-to-target path pattern used in pretraining is
therefore rarely present at positive new report sites. This does not bound
what static terrain could teach a future model; it does explain why learning
to propagate **prior report sources on these paths** gives little opportunity
in this queue. [Coverage audit](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_path_history_coverage.json).

## Reading and next data need

This experiment shows **synthetic spatial learning without useful report-site
transfer**. More steps on the same D8 pair task would improve its own target
at best; the 13/332 exposure check shows the relation is poorly matched to
the current report input. Bangkok's actual emerging water pattern also
depends on rainfall timing, drains, canals, pumps, barriers, and reporting
behavior, none of which are verified by a D8 terrain edge. The next valuable
pretraining target would be **time-matched observed urban wet/dry areas or
verified drainage connections**, with rain and infrastructure state aligned
to those observations. Existing transparent satellite pixels and unreported
cells cannot be relabeled as observed dry.

Q4 2025 has been examined repeatedly, so all model comparisons here are
exploratory. The result is a **report-site ranking**, not a flooding
probability or a validated heavy-rain simulation.

## Later target-mask transfer check

The fine-tuning above used naturally isolated sites but **did not train with
synthetically masked adjacent-history examples**. A later matched experiment
transferred these saved synthetic spatial weights into the exact 3 × 3
target-masked group CNN and compared it with the same LiDAR model trained from
scratch. Across three fine-tuning seeds, source-selected pretrained cell hits
were **62.78, 68.78, and 60.34 of 332**, versus matched scratch **53, 63, and
64** and distant report history **65**. The gain over scratch reversed in the
third seed, and mean pretrained capture remained below history. See the
[matched target-mask transfer report](FLOOD_CITYWIDE_PATH_PRETRAIN_TARGET_MASK.md)
for the exact design and per-fold artifacts.

## Reproduce

- [Pair generation, pretraining, and report fine-tuning](flood_citywide_path_contrast_pretrain.py).
- [Full-transfer five-fold metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_path_contrast_pretrain_isolated_fivefold.json) and [reset-head metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_path_contrast_pretrain_isolated_fivefold_reset_head.json).
- [Report-history path coverage code](flood_citywide_path_history_coverage.py) and [coverage artifact](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_path_history_coverage.json).
- Run `.venv/bin/python flood_citywide_path_contrast_pretrain.py`, then repeat with `--transfer-mode reset_head`.

The synthetic-pair audit checked 500 sampled pairs for exact distance,
elevation tolerance, connected reachability, and comparison non-reachability.
The frozen Q4 denominator remains **332** isolated positives and the distant
history row remains **65/332** cells and **44/158** groups.
