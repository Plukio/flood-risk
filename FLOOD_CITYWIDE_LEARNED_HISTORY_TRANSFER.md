# Learned report history: transfer across Bangkok districts

**Experiment run 22 October 2026.** The goal was to learn how report recency and nearby report patterns should affect a cell's daily rank, using **other districts and earlier dates**. The target remains a Traffy report-positive **250 m cell-day**. It is neither verified inundation nor a flood probability.

## Main result

The shared history models learned useful responses in some held-out places, but **none beat the fixed recent + nearby formula reliably across Bangkok's five geographic folds**.

| Daily top-10% queue | Fixed recent + nearby | Learned history, pairwise loss | Learned history, soft top-10% loss | History + static physical features |
| --- | ---: | ---: | ---: | ---: |
| Eight previously designated held-out districts; 978 Q4 positives | 60.23% | 63.38% | **63.69%** | 64.72% with pairwise loss; 59.40% with soft top-10% loss |
| Five disjoint 10-district holdout folds, pooled; 7,209 Q4 positives | **47.11%** | 46.24% | 47.10% | 46.96% with pairwise loss; 45.59% with soft top-10% loss |

**Read the rows separately.** The eight-district row ranks 6,579 candidate cells each day. The five-fold row uses a separate daily top-10% queue within each 10-district fold; its pooled capture is *not* one citywide top-10% queue. Each comparison within a row uses the same candidate cells, days and budget.

The eight-district soft-top-10 history gain over the fixed formula is **+3.47 percentage points**; its exploratory seven-day-block interval is **+1.51 to +6.19 points**. The corresponding five-fold pooled gain is **−0.01 points**, with interval **−0.97 to +1.00 points**. The pairwise model with physical features also improves the original eight-district group but loses slightly in the pooled five-fold test. Thus the first geographic split was unusually favorable.

| Five-fold Q4 split | Report-positive cell-days | Fixed formula | Soft-top-10 learned history | Difference |
| --- | ---: | ---: | ---: | ---: |
| Fold 1 | 1,284 | 51.87% | 53.89% | +2.02 points |
| Fold 2 | 1,570 | 54.20% | 56.50% | +2.29 points |
| Fold 3 | 1,065 | 35.49% | 33.24% | −2.25 points |
| Fold 4 | 1,767 | 43.60% | 44.13% | +0.53 points |
| Fold 5 | 1,523 | 48.00% | 44.82% | −3.18 points |

The learned models also rank some *first-in-a-year* report sites. Among 1,664 Q4 positive cell-days at cells with **zero own-cell reports in the preceding 365 days**, the fixed formula selects none in these fold queues; the soft-top-10 history model captures **1.38%** and the pairwise history model **5.23%**. That is a narrow gain on a sparse subgroup, alongside the overall trade-off. The cells may still have reports nearby; "first-in-a-year" does not mean the area was previously unobserved or newly flooded.

## What was learned

Each daily score uses reports **strictly before** that day. Inputs are local report counts over 7, 30, 90 and 365 days, plus report-count maps smoothed at **250 m, 750 m and 1,500 m**. Counts are log-transformed. The fixed benchmark uses only local 365-day and 90-day counts plus the 250 m-smoothed 365-day map, with weights **1, 0.4, 0.4**.

The learned history branch shares its weights across all cells and districts. We tested a positive-weight linear response and a 12-unit positive-weight hidden layer, both **monotone in report-history inputs**. The richer model adds a small static residual from terrain position, slope, mapped road/canal geometry and WorldCover built/crop/tree cover. It has no district identifier, future-day rain, or target-district labels in fitting. A hidden layer did not clearly beat the learned linear history model; the static branch did not transfer consistently.

Two losses tested the ranking task:

1. **Pairwise logistic:** each report-positive cell is compared with eight unreported cells from the same day: four sampled uniformly, two from the fixed formula's high-ranked cells, and two near its daily top-10% boundary. Unreported cells are *unknown*, not verified dry.
2. **Soft top-10%:** use 1,024 uniformly sampled unreported cells to estimate the day's 90th-percentile score, then penalize report-positive cells below that cutoff. It is a differentiable approximation to the actual queue; each day samples 64 positive cells, weighted by its full positive count.

Both models fit on **2023–2024** source-district observations. Checkpoints were selected by the respective surrogate loss on **January–September 2025** source districts. The **October–December 2025** target districts were scored afterward. For the original eight-district split, fitting excluded those districts' 2023–2024 positive labels and validation excluded their January–September 2025 labels. Every held-out district's earlier reports still enter its own history *features* at scoring time; this tests transfer of learned **rules**, not a cold start with no local history.

The pairwise validation loss kept improving while Q4 top-10% capture deteriorated in several places. The soft-top-10% loss aligned better with the queue on the original eight districts, yet still gave no pooled five-fold improvement. Lower training loss alone is therefore insufficient to select the operational ranker.

## Decision and limits

Keep the **fixed recent + nearby history formula** as the supported daily benchmark. The learned history model is a research candidate for the first-in-a-year subgroup and for identifying districts where shorter recency windows matter; it is not yet a replacement for the fixed queue.

The Q4 2025 period has been inspected repeatedly in this project, so even the fold results are exploratory. A genuinely new season and prespecified district folds are needed to establish improvement. Traffy reports indicate reported problem sites; unreported cells cannot establish false-positive rates or dry ground. These scores cannot simulate doubled rain or predict flood depth.

Reproduce the original split with [pairwise training](flood_citywide_learned_history_transfer.py) and [soft-top-10% training](flood_citywide_soft_top10_transfer.py). Reproduce the geographic folds with [pairwise folds](flood_citywide_learned_history_cv.py) and [soft-top-10% folds](flood_citywide_soft_top10_cv.py). The [pairwise metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_learned_history_fivefold_transfer.json) and [soft-top-10% metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_soft_top10_fivefold_transfer.json) include daily captures and paired intervals.
