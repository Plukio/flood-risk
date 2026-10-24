# Time-matched rain in the Bangkok reported-site CNN

**Exploratory run, 24 October 2026.** This experiment asks whether recent rain helps the spatial CNN choose locations of *new Traffy flood reports*. The outcome is still a reported-site proxy, not verified water extent or a flood probability. The 250 m site definition and Q4 2025 data were examined in prior work.

## Matched experiment

The 16-channel site CNN and a 21-channel version were trained on the **same 2023–2024 physical-site examples, comparison cells, folds, pairwise ranking loss, and optimizer schedule**. The extra five channels are `log1p` rain totals over the **1, 3, 6, 24, and 72 hours strictly before the decision cutoff**. Rain comes from the 50 cached [Open-Meteo historical hourly district-point series](https://open-meteo.com/en/docs/historical-weather-api) in `data/flood_bangkok_transfer/rain_*_2021_2025.json`. All cover 2021–2025, have `Asia/Bangkok` timestamps, and contain no null precipitation values. Each district-point estimate is broadcast over its district's 250 m cells; it is a coarse forcing layer, not street-scale observed rain.

The direct-category Traffy source files retain report timestamps. The site event's **first report timestamp** assigns it to the next-hour or next-six-hour target window. The timestamp is assumed to be Bangkok local time; that convention was not independently confirmed. Rain at or after the decision hour is excluded. The CNN's report-history channels still stop before the scored day, and the site medoid and first-in-year label are otherwise unchanged.

Five geographic folds train on source districts and score Q4 2025 in excluded districts. Every arm uses a **10% of cells per district per decision window** queue, with 20% of those slots reserved for zero-own-history cells. The learned arms use the same prespecified 50/50 percentile blend of nearby history and CNN score. The history queue is a common reference. Unreported comparison cells are unknown, not verified dry.

| Decision window | New Q4 sites | Nearby history | Matched CNN, no rain | Matched CNN, with rain | Rain minus no rain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Six hours | 471 | 5.68% | 7.64% | **8.34%** | **+0.70 points**, about 3.3 sites |
| One hour | 471 | 5.68% | **8.12%** | 8.04% | **−0.08 points**, about 0.4 site |

The six-hour rain gain is uneven: fold-level extra selected sites are **+4.44, +1.12, +0.01, −1.00, and −1.26**. The one-hour result does not repeat it. Among sites with at least **5 mm estimated in the preceding six hours**, six-hour capture is **6.98% with rain versus 4.16% without** across 86 sites, but the one-hour version reverses to **5.24% versus 7.20%** across 102 sites. These small subsets do not establish a stable intensity response.

![Rain-channel capture difference by held-out fold](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_hourly_rain_fold_delta.png)

## Interpretation

Adding historical rain is feasible and can alter which locations the CNN selects. **This test does not yet show a reliable predictive improvement**: the pooled six-hour gain is small, two folds lose, and the one-hour check is slightly worse. Rain close to report time is often low at the district proxy: only 18% of Q4 new sites have at least 5 mm during the six hours before a six-hour cutoff, versus 22% before their report hour. Local downpours, report delay, and the archive's coarse spatial scale can all weaken alignment.

The prior **10.01% daily source-selected site CNN** is a different queue design; its number cannot be treated as a matched no-rain control for this experiment. The 50/50 blend here was fixed before scoring. The one-hour run was added after reviewing the six-hour result, so both are exploratory rather than a fresh confirmatory test.

The historical Open-Meteo values are retrospectively estimated, not an archived real-time forecast. The next useful data improvement is **timestamped, finer-scale gauge or radar rain and independently observed flood onset/extent**, plus drainage operating state. A rain-conditioned reported-site ranker alone does not validate doubled-rain flood simulations.

## Reproduce

Run `.venv/bin/python flood_citywide_site_rain_hourly.py` for six-hour windows and `.venv/bin/python flood_citywide_site_rain_hourly.py --slot-hours 1` for one-hour windows. The [six-hour metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_hourly_rain_fivefold.json) and [one-hour metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_1h_rain_fivefold.json) contain fold results and training losses. The code is in [flood_citywide_site_rain_hourly.py](flood_citywide_site_rain_hourly.py).
