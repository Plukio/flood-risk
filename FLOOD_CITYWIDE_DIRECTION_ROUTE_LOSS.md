# Routing- and direction-aware CNN pretraining

**Tested 23 October 2026.** The previous synthetic route pretrain asked the
CNN to regress only total runoff arriving at a cell. That could reward generic
proximity without requiring the network to know which neighbor supplied the
flow. This experiment adds directional flow targets and a direct D8 downhill
direction task. All targets here are **generated from terrain**, not observed
urban water movement.

## Target and loss

For each synthetic rain/runoff scenario, the same cached BMA LiDAR D8 graph
routes source runoff through up to five downhill 250 m cells, with 0.72
retained at each step. At every cell the target is an **eight-component incoming
flux vector** `q[d]`, where `d` identifies the neighboring cell from which
water arrived. Its sum equals the earlier total-arrival target. Separately,
every terrain cell has a **nine-class outgoing label**: one of eight downhill
neighbors, or a sink with no eligible downhill neighbor. The source cell's
own runoff does not count as incoming flow at that cell.

The CNN has an eight-flux auxiliary head and a nine-class outgoing-direction
head. Its pretraining objective is fixed before the report evaluation:

```text
L = Huber(log1p(sum predicted incoming flux), log1p(sum target incoming flux))
  + 0.5 × sum over eight directions Huber(predicted flux[d], target flux[d])
  + 0.1 × cross-entropy(target incoming-flux share, predicted share)
  + 1.0 × cross-entropy(target outgoing D8 direction, predicted direction)
```

The incoming-share term uses only cells receiving meaningful synthetic flow;
the outgoing direction term uses all valid terrain cells, including sinks.
Each of 200 synthetic batches balances receiving and non-receiving centers.
For report fine-tuning, the synthetic heads are discarded, the static input
weights and spatial convolution blocks are transferred, and the report head
starts fresh. Fine-tuning uses the **same two-epoch pairwise report-ranking
loss and sampled examples** as the prior route experiment.

## Synthetic learning check

| Held-out terrain measure across five folds | Direction-aware result | Simple reference |
| --- | ---: | ---: |
| Outgoing downhill neighbor or sink, top-1 accuracy | **25.4–35.3%** | Most common class **14.8–18.8%** |
| Largest incoming-flux direction, top-1 accuracy on receiving cells | **20.2–25.7%** | Most common direction **17.1–19.0%** |
| Total-arrival Spearman against synthetic D8 target | **0.420–0.497** | No equal-reach distance control here |

The stronger loss makes the model learn some of the **synthetic D8 direction
rule** across excluded districts. Incoming directions remain much harder than
the local outgoing direction, and neither target validates actual stormwater
flow through Bangkok's drains and canals.

The first, incoming-only directional attempt used 100 batches and no direct
outgoing class. Its incoming direction accuracy was **16.5–18.2%**, roughly the
most-common-direction baseline. Its Q4 report-site result was **47/332**
isolated positive cells and **31/158** touching groups. That diagnostic led
to adding the outgoing-direction task. The final 200-batch result below uses
more synthetic training than the original 100-batch scalar experiment, so
their comparison is about observed outcomes, not a controlled compute ablation.

## Transfer to report-site ranking

The queue is unchanged: each Q4 2025 held-out district fold selects 10% of
candidate 250 m cells per day, with 20% of slots reserved for cells with no
own or adjacent prior-year report. These are **Traffy report-positive sites**,
not verified flooded cells.

| Method | Isolated positives found / 332 | Touching groups found / 158 |
| --- | ---: | ---: |
| Distant report history | **65 (19.58%)** | **44 (27.85%)** |
| LiDAR CNN trained from scratch | 48 (14.46%) | 30 (18.99%) |
| Scalar-arrival route pretrain, 100 batches | 50 (15.06%) | 31 (19.62%) |
| **Direction-aware route pretrain, 200 batches** | **46 (13.86%)** | **32 (20.25%)** |

The directional model finds one more touching group than the scalar-pretrained
CNN, but four fewer positive cells. It remains well below distant report
history on both measures. On source-district January–September 2025
validation it found **459.9 of 3,508** isolated positive cells (13.11%),
versus **463.18** (13.20%) for distant history. There is no report-ranking
benefit to retain from this loss.

The learned direction is a **terrain-only rule**. Mapped canal geometry was
available as a CNN input but did not determine the synthetic routing target;
pipe connectivity, barriers, pump/gate operation, and tide-controlled
direction were unavailable. Also, Q4 2025 has been inspected repeatedly,
so this is an exploratory comparison rather than a fresh prospective test.

## Reproduce

- [Direction-aware pretraining and fine-tuning](flood_citywide_direction_route_pretrain.py).
- [Five-fold metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_direction_route_outgoing_pretrain_isolated_fivefold.json).
- [Final CNN weights](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_direction_route_outgoing_pretrain_isolated_fivefold.pt).
- [Initial incoming-only diagnostic](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_direction_route_pretrain_isolated_fivefold.json).
- Run `.venv/bin/python flood_citywide_direction_route_pretrain.py`.

The synthetic edge-direction checks, loss-gradient check, and unchanged queue
denominators passed. Unreported cells remain unknown, not verified dry.
