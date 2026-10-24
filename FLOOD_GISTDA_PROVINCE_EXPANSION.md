# Province expansion: does more training help Bangkok?

Run on 22 October 2026. The target here is **GISTDA mapped-water area**, ranked across 250 m cells. A cell without a polygon is **unknown**, not verified dry. These numbers are not flood probabilities or road-flood detection rates.

## Data and fixed evaluation

- Scanned 21 candidate province patches and added 15 with sufficient GISTDA polygon coverage. Together with the original five, training uses **20 nonoverlapping 20 × 20 km patches across 17 provinces**. The Nong Chok patch is withheld from fitting, normalization, and epoch selection.
- All stages use the same **eight GISTDA-positive dates in September–October 2026** as the Nong Chok test. The Bangkok-only result is the part of that patch inside the BMA boundary: **5,001 candidate cells, 308.8 km²**.
- Inputs are [SRTM elevation](https://registry.opendata.aws/terrain-tiles/) and slope, [OpenStreetMap](https://www.openstreetmap.org/) road/canal proximity and density, [Open-Meteo historical rainfall](https://open-meteo.com/en/docs/historical-weather-api), month, and six [ESA WorldCover 2021](https://esa-worldcover.org/en) land-cover fractions. Labels come from [GISTDA daily flood products](https://data.go.th/dataset/flood-disaster-data). The CNN ranks the distribution of known mapped-water area within each patch. The 2026 training set uses five dates and selects epochs on two later dates from the *training* provinces.
- A second experiment adds six spaced 2025 GISTDA dates from the same external patches. Empty products create no dry examples. Those 2025 dates were chosen after earlier inspection of the 2026 Bangkok test, so this is an exploratory sensitivity check.

**Metric:** fraction of mapped-water area captured in the top-ranked 10% of candidate cells on each date, averaged with mapped area as the weight. Random ranking would capture 10% in expectation.

| External patches | Positive training patch-days | Full Nong Chok patch | Inside BMA boundary |
| --- | ---: | ---: | ---: |
| 5, original 2026 training | 23 | 30.72% | 34.55% |
| 10, 2026 training | 46 | 33.85% | 37.95% |
| **15, 2026 training** | 71 | **34.98%** | **40.60%** |
| 20, 2026 training | 90 | 34.15% | 39.97% |
| 5, plus 2025 dates | 41 | 28.78% | 32.69% |
| 15, plus 2025 dates | 127 | 34.67% | 41.34% |
| 20, plus 2025 dates | 153 | 32.76% | 38.87% |
| **WorldCover crop fraction alone; no fitting** | — | **37.14%** | **44.32%** |

The best 2026 CNN gains **6.05 percentage points** inside Bangkok over the original five-patch CNN, and wins on seven of eight dates. Adding the last five patches lowers its score. Adding 2025 data raises the 15-patch Bangkok score by only **0.75 points** and lowers the 5- and 20-patch scores. The crop-only rule remains **2.98 points ahead** of the best CNN and wins on seven of eight dates. Eight dates from one season are a small, correlated test set.

## Does that transfer to urban flood reports?

As a separate diagnostic, the fixed CNNs were scored against **107 Traffy flood-report-positive cell-days on 24 days in October–December 2025**, inside the same Bangkok area. This applies 2026-trained models **backward to 2025** and changes the target from satellite mapped water to reports; it is not a forward-time validation. The report-history rule uses January–September 2025 only.

| Ranking for top 10% of cells each day | Q4 2025 report-positive cell-days captured |
| --- | ---: |
| 2025 report history | 30.40% |
| WorldCover built fraction | **33.64%** |
| WorldCover crop fraction | 1.87% |
| 15-patch 2026 GISTDA CNN | 0.93% |
| 15-patch GISTDA CNN with added 2025 dates | 4.67% |

The GISTDA-trained ranking concentrates on mapped agricultural water, while complaints concentrate in built areas. The small report sample and backward-time comparison limit the estimate, but it clearly does **not** show a useful urban-report ranking. A GISTDA map score should not be presented as a Bangkok street-flood score.

## Decision and limits

The province expansion helps the **mapped-water** ranking up to about 15 external patches, then levels off. The basic land-cover rule still wins. More examples of the same GISTDA product have not solved the urban-target mismatch. A meaningful next training set needs dated observations of water on built surfaces or roads across multiple cities, with entire cities and later seasons held out. Report absence still cannot stand in for verified dry ground.

The rainfall feature is a full local-day total, which can include rain after the satellite acquisition time. These backtests therefore do not validate a forecast or a “twice as much rain” scenario. There are no calibrated flood probabilities or measured water depths in this experiment.

## Artifacts and reproduction

- Selection and source audit: [`province_candidate_scan_2026.json`](artifacts/flood_gistda_multiregion/province_candidate_scan_2026.json), [`province_expansion_data_audit.json`](artifacts/flood_gistda_multiregion/province_expansion_data_audit.json), [`province_2025_extra_data_audit.json`](artifacts/flood_gistda_multiregion/province_2025_extra_data_audit.json).
- Model and Bangkok metrics: [`province_expansion_bangkok_metrics.json`](artifacts/flood_gistda_multiregion/province_expansion_bangkok_metrics.json), [`province_expansion_2025_augmented_bangkok_metrics.json`](artifacts/flood_gistda_multiregion/province_expansion_2025_augmented_bangkok_metrics.json), [`province_expansion_bangkok_bma_only.json`](artifacts/flood_gistda_multiregion/province_expansion_bangkok_bma_only.json).
- Separate report stress check: [`province_expansion_traffy_2025_stress.json`](artifacts/flood_gistda_multiregion/province_expansion_traffy_2025_stress.json).

After the data caches are built, rerun the fitting and evaluations with:

```bash
.venv/bin/python flood_gistda_province_expand_train.py
.venv/bin/python flood_gistda_province_2025_extra_train.py
.venv/bin/python flood_gistda_province_expand_bangkok_eval.py
.venv/bin/python flood_gistda_province_expand_traffy_stress.py
```

The full source-data build uses `flood_gistda_province_expand_data.py` and `flood_gistda_province_2025_extra_data.py`; both require the public GISTDA catalog and other open data endpoints. The earlier [six-region geographic backtest](FLOOD_GISTDA_LANDCOVER_BACKTEST.md) records why crop cover is the required baseline.
