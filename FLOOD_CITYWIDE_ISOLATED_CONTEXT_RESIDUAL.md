# Can wider spatial context improve isolated-site ranking?

**Tested 23 October 2026.** The previous isolated-site CNN read raw report history only within a seven-cell (about 1.75 km) learned radius, while the winning simple score used report smooths with 750 m, 1.5 km and 3 km scales. This test supplies those **wider pre-day history maps as CNN input channels** and trains the CNN to correct, rather than replace, the simple score.

## Result

The added spatial input **did not produce a transferable gain**. The correction strength chosen from January–September 2025 **source districts** captured fewer isolated Q4 groups than the unchanged score in all three seeds. Its isolated cell-day capture was higher in one seed, lower in one, and tied in one. A small correction fixed at 0.1 captured a few additional cell-days, but group capture was lower or tied; this is a diagnostic on a repeatedly inspected Q4 set, not a newly validated improvement.

All rows use the same five held-out district folds, a daily budget of 10% of candidate cells per fold, and **20% of inspection slots reserved for cells with no own or adjacent report in the preceding 365 days**. There are **332 isolated positive cell-days** and **158 fold-scoped same-day touching groups**. A group counts when at least one member is selected; fractional top-k ties have fractional expected captures.

| Isolated-slot score | Isolated cell-days captured, seeds 0 / 10000 / 20000 | Isolated groups captured, same seeds | Total positive cell-day capture |
| --- | ---: | ---: | ---: |
| Fixed distant-history score, correction 0 | **65 / 65 / 65** of 332 | **44 / 44 / 44** of 158 | 42.258% |
| CNN correction fixed at 0.1, exploratory | 66 / 68.78 / 68 | 41 / 42.78 / 44 | 42.272–42.311% |
| CNN strength selected on source validation | 66 / 59 / 65 | 38 / 40 / 42 | 42.175–42.272% |

The earlier **no-reserve** report-history queue captured **47.113%** of all positive cell-days and zero isolated positives on this fold-based task. The reserve is an explicit exploration trade-off; neither CNN result changes it.

## What the CNN learned from

For each day, the input has 21 spatial layers: four strictly pre-day 7/30/90/365-day report-history maps; ten source-normalized terrain, road, canal and land-cover features; a candidate-area mask; five strictly pre-day distant report-history smooths; and a grid-validity mask. The added smooths cover 365-day reports at Gaussian standard deviations of **750 m, 1.5 km and 3 km**, plus 90-day reports at 750 m and 1.5 km. The CNN's five spatial steps combine these layers within a 15 × 15 cell patch; the wide smooths bring in information beyond that patch. A patch/full-grid equality check passed.

The CNN's output is a zero-initialized residual on the fixed distant-history score. It is trained on 2023–2024 **actual isolated positive groups**, sampling one representative per same-day touching group and comparing it with unreported isolated candidates through a pairwise loss. Unreported comparisons are **unknown**, not verified dry. The objective regularizes the residual and uses no future rain or scored-day report.

Candidate correction strengths were **0, 0.1, 0.25, 0.5 and 1**. Each fold chose the strength with the best isolated **group capture** in January–September 2025 source districts, breaking ties by cell capture and then smaller strength. Q4 2025 excluded-district outcomes were consulted only afterward. Source selection still often chose medium or large corrections that transferred poorly. This suggests a **geographic/seasonal selection mismatch**; the current design does not separate the two shifts.

## Next experiment

If using only current Traffy labels, the next credible model selection is **nested geographic validation**: withhold some districts from both weight fitting and correction-strength selection inside each outer training fold, then refit on all outer-source districts before scoring the outer holdout. Train and score **same-day positive groups** directly, rather than scoring one random member of each group, so the training target matches the site-level inspection objective. Keep the fixed distant-history score as the zero-correction control and use a new season or new city for confirmation; Q4 2025 has already been inspected many times.

More spatial layers alone cannot establish that the model understands water flow. Traffy reports reflect reporting behavior and give no verified dry cells. Prior satellite auxiliary training did not show a stable urban-report gain. For physical flood extent or rainfall counterfactuals, the stronger data need is timed urban water observations, drainage connectivity, and pump/gate state; a canal line or static satellite map by itself does not supply those labels.

## Reproduce

Run [`flood_citywide_isolated_context_residual.py`](flood_citywide_isolated_context_residual.py) with no arguments and then `--seed-offset 10000` and `--seed-offset 20000`. The [first](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_context_residual_fivefold.json), [second](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_context_residual_fivefold_seed_10000.json), and [third](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_context_residual_fivefold_seed_20000.json) artifacts contain per-fold source validation, selected strengths, and daily held-out capture. Matching `.pt` files contain the CNN weights.
