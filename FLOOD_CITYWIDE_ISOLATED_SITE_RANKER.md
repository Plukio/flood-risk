# Training on actually isolated Bangkok report sites

**Run 23 October 2026.** This experiment asks whether a ranking head trained on actual report sites with no recent nearby report can transfer to excluded Bangkok districts. An eligible 250 m cell has **no Traffy flood report in its own cell or any of its eight adjacent cells during the previous 365 days**. This is report isolation, not proof that water has never flooded there.

## Result

The separate CNN does **not** improve isolated **cell-day** capture over a simple distant-report-history score. With the same daily queue budget, the simple score captures **65 of 332** isolated Q4 2025 positive cell-days (**19.58%**). Across three training seeds the CNN captures an expected **54.34, 58, and 60.34** (**16.37–18.18%**). The tabular model captures **26–27.6** (**7.83–8.31%**). Fractional counts arise from allocating exact top-k tie weights; they are expected captures, not fractional complaints.

Each day, the queue covers **10% of candidate cells within each held-out district fold**. For the last three rows, **20% of those slots** are reserved for cells meeting the stricter isolated definition; all remaining cells compete for the other 80% using the fixed recent + nearby history score. Each method has the same reserve and candidate set. The five folds together contain 7,209 Q4 positive cell-days, including 332 isolated positives. Touching isolated positives form **158 fold-scoped same-day groups**; a group counts as captured when at least one member is selected, with fractional boundary ties counted at their allocation weight. The group count exceeds the 149 citywide groups because district-fold boundaries can split a group.

| Isolated-slot score | All positive cell-days captured | Isolated positive cell-days captured | Isolated touching groups captured |
| --- | ---: | ---: | ---: |
| Fixed recent + nearby history, **no reserve** | **47.11%** | 0 / 332 | 0 / 158 |
| **Distant report history** | 42.26% | **65 / 332 (19.58%)** | 44 / 158 (27.85%) |
| Tabular gradient boosting, three seeds | 41.72–41.74% | 26–27.6 / 332 (7.83–8.31%) | 20.6–21 / 158 (13.04–13.29%) |
| Spatial CNN, three seeds | 42.11–42.19% | 54.34–60.34 / 332 (16.37–18.18%) | 37.34–46 / 158 (23.63–29.11%) |

The CNN's group score is above the simple rule in **one of three** seeds and below it in two. Its cell-day score is below the simple rule in **all three**. The no-reserve history queue remains strongest for **total** report capture; allocating isolated slots reduces total capture by about **4.9–5.0 percentage points** in this experiment. That is a real discovery-versus-total-capture trade-off, not an operational win.

## Training and holdout

The 2023–2024 training period contains **1,731** isolated positive cell-days but only **767 same-day touching groups** citywide. Training samples one representative cell per touching positive group on each date, rather than counting neighboring cells in a 100 m complaint buffer as independent examples. Each representative is compared with same-day **unreported isolated candidate cells** sampled both randomly and from high distant-history scores. The comparison cells are **unknown**, not verified dry; same-day source-district positives and their neighboring cells are excluded from comparison sampling. Outcomes in target districts never enter fitting or comparison selection. Earlier reports from those districts remain legitimate pre-day inputs.

The CNN is the existing five-step `SpatialNovelCNN` over a 15 × 15 grid patch, trained only on these actual isolated positives with a pairwise ranking loss. It reads strictly pre-day 7/30/90/365-day report counts, SRTM terrain, OSM road/canal geometry, WorldCover built/crop/tree cover, and the candidate mask. The gradient-boosted comparator receives the same physical features plus wider pre-day history summaries. Neither model receives actual future rain or 2026 satellite outcome labels.

The fixed simple score is `0.6 log1p(Gaussian_750m(prior 365d reports)) + 0.3 log1p(Gaussian_1500m(...)) + 0.1 log1p(Gaussian_3000m(...))`, using 250 m cells and Gaussian standard deviations of 3, 6, and 12 cells. For an isolated candidate, own and adjacent prior-year counts are zero, so this score gets its information from farther away.

Five geographic folds train on 40 districts in 2023–2024, examine January–September 2025 **source districts** for validation, and score Q4 2025 in 10 excluded districts per fold. The source-validation CNN group score exceeds the simple score in **14 of 15 fold-seed comparisons**, but its held-out Q4 group advantage is inconsistent. The difference may reflect district, seasonal, and reporting changes; this design does not separate those causes or test transfer to another city. Q4 2025 has also been inspected repeatedly in prior project work, so this is exploratory rather than a fresh confirmatory holdout.

## Decision

For an isolated-site exploration queue, retain the **distant-history score** as the current benchmark. Direct training on isolated positives removes the synthetic-repeat-site shortcut, but these 767 grouped training examples are insufficient to show a transferable CNN gain. The CNN's source-validation advantage is a warning against choosing a model only from same-area validation.

The next training improvement should add **new, independently located urban water observations** or genuinely later report seasons, then repeat the same isolated-site queue without changing the budget or Q4-informed model choices. BMA road-flood records currently provide only five approximately located isolated Q4 site-days, too few for a meaningful confirmation. Satellite mapped water is a different target and cannot be treated as a dry/positive complaint label. This experiment does not validate flood probabilities, physical flow, or doubled-rain scenarios.

## Reproduce

Run [`flood_citywide_isolated_site_ranker.py`](flood_citywide_isolated_site_ranker.py) with no arguments and then with `--seed-offset 10000` and `--seed-offset 20000`. The [first-seed metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_site_ranker_fivefold.json), [second-seed metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_site_ranker_fivefold_seed_10000.json), and [third-seed metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_isolated_site_ranker_fivefold_seed_20000.json) include per-fold counts, source validation, daily isolated captures, and training audits. Matching `.pt` files contain the CNN weights.
