# BMA pipe-network feature in the Bangkok report model

**Exploratory test, 23 October 2026.** We tested whether the newly found BMA
drainage GIS improves the existing BMA-road-pretrained, target-masked CNN's
ranking of *future Traffy report-positive cells*. This is a report-site queue,
not a measured flood-extent or rainfall-simulation test.

## Source and constructed network

The [public BMA drainage service](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer?f=pjson)
returned 81,574 pipe polylines, 3,496 canal polylines, 273 pump-equipment
points and 77 BMA gate points. The pipe layer's `YEAR=2564` describes a 2021
compilation; the service has no dated operating state. We cached the exact
[downloaded snapshot](artifacts/flood_open_weak_train/multi_target/citywide/bma_drainage_citywide_2021_raw.json.gz)
and [network audit](artifacts/flood_open_weak_train/multi_target/citywide/bma_drainage_citywide_2021_network_audit.json).

We used 81,276 single-path pipes with valid geometry. Endpoints within 5 m were
snapped into candidate graph nodes; **a crossing in the middle of a line was
not connected automatically**. Of 8,445 pipe records whose `TO_ROAD` names a
canal or river, 6,656 last endpoints lie within 20 m of a mapped canal and
4,071 also match the nearest canal's normalized name. These 4,071 endpoints
are *candidate links*, not confirmed operating outfalls. The graph has 99,113
snapped nodes. We measured undirected shortest path distance through mapped
pipes to the closest candidate endpoint. Pump and gate locations enter as
proximity, not as assumed on/off states.

At each 250 m cell, the added static channels are `exp(-nearest pipe/250 m)`,
`exp(-network path to candidate canal endpoint/1 km)` when within 250 m of a
pipe, and `exp(-nearest pump/500 m)` and `exp(-nearest gate/500 m)`. Path length
is approximated at the nearest pipe by its closer endpoint. Cells farther
than 5 km through the network receive the capped low path score. The control
relocates the same number of candidate endpoints to random pipe ends within
each pipe district, leaving pipe, pump and gate geometry unchanged. Its
candidate reach can still vary with component size.

Only **4,603 of 25,812** modeled Bangkok cells have a real candidate link
within 5 km and are within 250 m of a pipe. That includes **69 of 332** Q4
strict isolated report-positive cell-days. The broader pipe-proximity channel
covers about 14,752 candidate cells within 250 m. The missing reach is a
substantial limit on what this candidate-link feature can change.

## Matched model and held-out result

The prior BMA-road model is frozen. A three-convolution static CNN learns a
correction from source-district 2023–2024 strict isolated Traffy groups,
using the same group-vs-unreported pairwise loss. It sees source-normalized
relative LiDAR elevation, OSM canal distance and built cover together with
the four drainage channels. **No report history enters the new static branch.**
The frozen main CNN keeps its exact 3 × 3 target-history mask, including the
Gaussian-history correction. Source districts in January–September 2025
select correction strength from `0, 0.1, 0.25, 0.5, 1`; October–December 2025
districts are held out in the same five geographic folds as the BMA-road test.
Every queue uses 10% of candidate cells per day and reserves 20% of slots for
strict isolated candidates. The shuffled-endpoint arm has the same model,
training samples, seeds and selection rule.

| Fine-tuning seed | Frozen BMA model, cells / groups | Real drainage, cells / groups | Shuffled endpoints, cells / groups |
| --- | ---: | ---: | ---: |
| 0 | 70 / 44 | 71 / 45 | 70 / 43 |
| 10,000 | 71.78 / 51.78 | 68.78 / 48.78 | 71.78 / 51.78 |
| 20,000 | 64.78 / 47.78 | 61 / 44 | 54.06 / 38.06 |
| **Mean of repeated fits on the same Q4** | **68.85 / 47.85** | **66.93 / 45.93** | **65.28 / 44.28** |

The denominator is **332** isolated report-positive cell-days and **158**
touching positive groups, not all flooded cells. Fractional hits come from
the queue's boundary-tie allocation. The mean real-drainage cell capture is
**20.16%**, versus **20.74%** for the frozen BMA model; group capture is
**29.07%** versus **30.29%**. The real link placement is better than shuffled
on average, but **the drainage adapter does not improve the frozen model
consistently**. Two of three seeds lose cells and groups against it. These
three fits reuse the same Q4 dates and do not form three independent tests.
Source-only selection still sometimes picks a correction that fails in the
excluded districts.

This is **not a GNN trained on verified hydraulic edges**. It gives a CNN a
graph-derived static path feature. Snapping digitized endpoints within 5 m,
attaching cells to the nearest pipe without inlet points, and identifying an
outlet by a road-name field and canal proximity can all be wrong. There is no
observed flow direction, pipe blockage, gate opening, pump operation, or
inside/outside canal head for these report dates. Unreported comparison cells
are unknown rather than verified dry. Q4 2025 was inspected in earlier model
iterations, so this is exploratory even with district holdout. The result
does not validate a flood probability, depth, or doubled-rain scenario.

**Decision:** keep the frozen BMA-road model and the simpler history queues as
the current comparators; do not promote this GIS adapter on these results.
The most valuable next data would verify inlet-to-pipe and pipe-to-outfall
links, operating state at event time, and urban wet/dry observations.

## Reproduce

- Build the [network features](flood_citywide_drainage_network.py):
  `.venv/bin/python flood_citywide_drainage_network.py`.
- Run the [matched adapter](flood_citywide_drainage_adapter.py) with seed
  offsets `0`, `10000`, and `20000`.
- Generate the [three-seed summary](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_bma_drainage_adapter_three_seed_summary.json)
  with `.venv/bin/python flood_citywide_drainage_summary.py`.
