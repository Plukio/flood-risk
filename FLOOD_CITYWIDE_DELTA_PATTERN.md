# Where the report model gains or loses against nearby history

**24 October 2026.** This follows the [50-district coverage curve](FLOOD_CITYWIDE_DATA_COUNT_CAPTURE_CURVE.md). The comparison is the same Q4 2025 **first-in-prior-year Traffy report-site** task, with a separate daily top-10% cell budget in each district and 20% of queue slots reserved for novel cells. The baseline is nearby report history; the model is the geographically held-out masked CNN plus source-selected ordinary-neighbor score propagation. Outcomes are report-positive **cell-days**, not verified inundation.

![District uplift and proximity breakdown](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_novel_delta_pattern.png)

## Size and location of the uplift

Across 50 districts, the model wins in **28**, loses in **15**, and ties in **7**. It captures an expected **36.28 more** of the **1,664** novel-positive cell-days than nearby history at identical per-district budgets (10.06% versus 7.88%). An individual small-district percentage can move sharply with one or two sites; the table below uses places with at least 20 Q4 novel-positive cell-days.

| District | Pre-Q4 report-positive days per 100 cells | Q4 novel positives | Extra sites captured by model | Uplift |
| --- | ---: | ---: | ---: | ---: |
| Bang Khun Thian | 38.69 | 123 | **+13.00** | +10.57 points |
| Lat Krabang | 21.87 | 81 | **+5.06** | +6.25 points |
| Nong Chok | 10.41 | 52 | **+4.00** | +7.69 points |
| Bang Phlat | 100.00 | 33 | **−4.08** | −12.36 points |
| Phasi Charoen | 33.44 | 36 | **−4.00** | −11.11 points |
| Bang Na | 153.75 | 28 | **−2.66** | −9.50 points |

Raw **report count** has essentially no district-level association with the uplift (Spearman **−0.067**). Report **density per candidate cell** is more informative descriptively (Spearman **−0.361**). The lower-density 25 districts gain **+3.77 points** in pooled capture; the higher-density 25 are effectively flat at **−0.10 points**. The lower-minus-higher gain interval from resampling districts is **+0.47 to +7.47 points**. Removing the largest winning district, Bang Khun Thian, leaves **+2.80 points** in the lower-density group. This density split was examined **after seeing Q4**, alongside several candidate features; it is a hypothesis, not a confirmed rule. Phasi Charoen is a clear low-density counterexample.

![Uplift versus report density](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_novel_delta_vs_report_density.png)

## What kind of new sites does the model find?

For each Q4 positive site, count the shortest side-neighbor steps to **any cell reported during the prior 365 days**, using only reports dated before that Q4 day. A side step is 250 m. This is a report-history connection, **not a proven water-flow path**.

| Pre-day history reach of a novel positive | Positives | Baseline captured | Model captured | Model uplift |
| --- | ---: | ---: | ---: | ---: |
| Adjacent, one side step | 1,170 | 131.06 (11.20%) | 163.34 (13.96%) | **+32.28 sites** |
| Two or three side steps | 415 | 0 | 3 (0.72%) | +3 sites |
| Beyond three side steps | 79 | 0 | 1 (1.27%) | +1 site |

**About 89% of the net extra captures are adjacent to a recently reported cell.** In lower-density districts, the model captures **103.4 of 591** adjacent positives versus **70.44** for nearby history; in higher-density districts it captures **59.94 of 579** versus **60.62**. That is consistent with the CNN better sorting some sparse report fringes while the simple nearby score remains competitive where reports are dense. It does **not** show broad discovery of isolated new flooded sites: the model catches only **1 of 79** positives beyond three side steps. Nong Chok is a small exception: its four extra captured positives are two or more steps from a prior report cell.

The [Traffy target construction](flood_multi_target_data.py) marks every 250 m cell touched by a **100 m complaint buffer**. A new cell next to a prior positive cell can therefore be a nearby or shifted report of a known physical location; this site-level check cannot distinguish that from a genuinely new flood location. Earlier 2025 Q4 experiments were repeatedly inspected, and the district-density split and feature correlations are exploratory. WorldCover built/crop fractions at the Q4 positives show only weak district-level association with uplift (**+0.120** and **+0.209** Spearman respectively); they do not explain the winners by themselves.

The defensible interpretation is that the present model mainly **reorders report-prone edges near known sites**. It has not learned a reliable transferable rule for previously unreported, distant flooding. A stronger next test would group raw Traffy coordinates into physical sites before the first-time label, then repeat this proximity breakdown on a new season and compare the same per-district inspection budgets.

Reproduce with [the analysis script](flood_citywide_delta_pattern.py). The [district CSV](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_novel_delta_pattern_districts.csv), [positive-site CSV](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_novel_delta_pattern_positive_sites.csv), and [result JSON](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_novel_delta_pattern.json) contain the counts and per-site queue weights. The script checks every district's hits against the prior two-line curve.
