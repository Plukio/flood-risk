# Satellite surface cover on Bangkok flood reports

Run on 22 October 2026. This uses the [existing Bang Kapi/Bang Khae report model](FLOOD_BANGKOK_NEW_FEATURES.md) to test the concrete-surface question on an urban target. A positive is a **Traffy flood-report buffer touching a 250 m cell on a day**. Missing reports are sampled comparisons, not verified dry observations; the model predicts reporting as well as flooding.

The existing `old_cover` group already includes [ESA WorldCover 2021](https://esa-worldcover.org/en) **built, combined vegetation, and water** fractions plus OSM building and road features. `split_cover` keeps those inputs and separately adds WorldCover **crop, tree/grass, bare ground, and wetland** fractions. Both use identical report rows, training samples and weights, tree settings, and October–December test periods. The [extraction audit](data/flood_bangkok_transfer/worldcover_split_250m_audit.json) and [matched model metrics](artifacts/flood_bangkapi_surface/surface_cover_split_backtest_metrics.json) preserve the check.

| Test | Old cover: cell Spearman / daily AP | Split cover: cell Spearman / daily AP |
| --- | ---: | ---: |
| 2025 Bang Kapi → Bang Khae | **0.276 / 0.03055** | 0.272 / **0.03119** |
| 2024 Bang Kapi → Bang Khae | **0.209 / 0.02184** | 0.204 / 0.02004 |
| 2025 Bang Khae local | 0.353 / 0.02966 | **0.355 / 0.03341** |
| 2024 Bang Khae local | 0.334 / **0.01809** | **0.337** / 0.01631 |

`Cell Spearman` ranks Bang Khae cells by summed predicted report-positive days against observed report-positive days. `Daily AP` ranks all cell-days. The split classes have **no consistent benefit**, and both transferred cell rankings worsen slightly. The 2021 WorldCover built fraction still looks relevant: across 2022–25, Bang Khae report-positive cell-days average **72.5% built cover**, versus **59.4%** for all included cells. Built fraction alone ranks 2025 report counts with Spearman **0.273** and 2024 with **0.189**. This association also reflects where people and roads generate reports; it does not prove that concrete caused a given flood.

The practical choice is to **keep the existing built/vegetated/water surface features** and retain separated classes only as a documented candidate. For runoff, a stronger feature would count connected paved area *uphill of each cell* and combine it with local drainage, road, and rain state. The [six-region GISTDA land-cover test](FLOOD_GISTDA_LANDCOVER_BACKTEST.md) is a different target: 85.2% of its mapped-positive area lies in mostly cropped cells, so it cannot validate paved-street runoff.

Reproduce with `.venv/bin/python flood_bangkok_worldcover_split.py` followed by `.venv/bin/python flood_bangkok_cover_split_backtest.py`. The 2024 run uses a 2025 OSM snapshot and both test seasons have been examined in earlier work, so this remains exploratory.
