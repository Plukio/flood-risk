# Replacing "new cell" with a first-in-year reported site

**Evaluation update, 24 October 2026:** The 20% discovery-slot reserve below
was an illustrative allocation, not a measured BMA policy or an allocation
selected for total capture. For the current paper's main result, use the
[reserve-free ranking analysis](FLOOD_SITE_QUOTA_FREE_RANKING.md): it selects
settings by source-district mean rank percentile and reports capture across
fractions of unfamiliar candidate cells. The figures below remain conditional
on the fixed reserve and should not be read as full-queue uplift.

**24 October 2026.** This experiment addresses a specific weakness in the
[district uplift audit](FLOOD_CITYWIDE_DELTA_PATTERN.md): a Traffy complaint's
100 m buffer can mark several 250 m cells, making an adjacent cell look like a
new flood location. The target here is a **dated cluster of raw direct-category
Traffy points**, evaluated by its representative 250 m cell. It remains a
report-site proxy, not verified flood extent.

## Result

The queue visits the highest-ranked **10% of cells per district per day**.
Twenty percent of those slots are reserved for cells with no own-cell report
in the preceding 365 days; all scores use the same slots. A physical site is
"new" when no report point from an earlier day in the preceding 365 days is
within the tested radius of any point in its same-day cluster. Complete-linkage
clustering limits each same-day group's diameter. The medoid report point
selects the cell to check.

| New-site radius | Q4 new site events | Nearby history | Earlier masked CNN + paths | Site-trained, source-selected CNN |
| --- | ---: | ---: | ---: | ---: |
| 100 m | 1,153 | 14.16% | 14.97% | **15.12%** |
| **250 m** | **471** | **5.68%** | **7.10%** | **10.01%** |
| 500 m | 112 | 0.00% | 2.68% | **6.25%** |

The 250 m result corresponds to **47.16** expected selected site events for the
site-trained model, versus **26.74** for nearby history and **33.42** for the
older CNN. Fractional counts arise from equal-score boundary ties. The model
adds **+4.34 percentage points** over history, or about 20 events. District
resampling gives a paired interval of **+0.82 to +8.21 points**; seven-day
block resampling gives **+0.51 to +8.59**. Against the older CNN the gain is
**+2.92 points**, with intervals **−1.13 to +7.28** by district and
**−1.56 to +6.64** by seven-day blocks. These descriptive intervals omit
target-design and model-selection uncertainty; Q4 2025 was repeatedly
inspected during the project.

### Prior report density × new-site distance

The matrix groups 50 districts into **five density bands of ten districts**
by report-positive cell-days per 100 candidate cells during the 365 days
before Q4. It holds **same-day site grouping fixed at 250 m**, then includes
only sites whose nearest earlier report is beyond 100, 250, or 500 m. The
rows are therefore nested subsets of the same site events. Every square
shows the percentage-point lift, history → CNN capture, ratio when meaningful,
and number of site events.

![Density by new-site-distance performance matrix](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_density_distance_matrix.png)

At **250 m**, the lowest-density band improves from **9.0% to 16.5%** across
133 new sites (**1.8×**). The highest-density band improves from **3.8% to
11.4%** across 65 sites (**3.0×**). The highest-density baseline has only
**2.46 expected hits**, so the ratio is sensitive to a few events. The
middle-density band **falls from 7.5% to 5.7%** across 70 sites. This is not
a monotone density effect.

Among sites more than **500 m** from prior reports, the lowest-density band
rises from **0% to 12.3%** across 57 sites. Other density bands are sparse:
the highest contains only **three** such sites. An improvement over zero has
no finite "times better" ratio. District density is a broad reporting
measure, not local drainage or local prior-site distance. These finer bands
are descriptive Q4 analyses, not proof that density changes model
effectiveness.

![District-level 250 m site uplift versus prior report density](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_250m_lift_by_district_density.png)

The district plot shows the variation hidden by the density bands. Each dot
is one district, sized by its number of Q4 new sites. Bang Khun Thian alone
accounts for **9 of the roughly 20 extra selected 250 m sites** citywide.
There is no smooth or monotone district-density relationship. The
[district-level CSV](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_250m_lift_by_district_density.csv)
records every district's denominator and expected hits.
The [matrix script](flood_citywide_physical_site_density_matrix.py),
[nine-cell CSV](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_density_distance_matrix.csv),
and [underlying new-site events](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_density_distance_events.csv)
allow each square to be checked.

The source-selected queue captures **35.09% of all 2,326 Q4 site events**, versus
**34.78%** for nearby history. Thus the new-site gain does not produce a large
change in total site-event capture. For sites **250–750 m** from the nearest
earlier report, capture is **10.78%** versus **6.25%** for history. Only **43**
events are more than 750 m from a prior report; the new model catches **one**,
the earlier CNN two, and history none. Discovery of truly isolated sites is
still weak.

## What was trained

Both new arms use the existing spatial CNN inputs: pre-day 7/30/90/365-day
report maps, terrain, OSM roads and canals, and WorldCover. The output is a
learned correction to the **nearby-history score**. Positives are 2023–2024
first-in-year **250 m physical-site** medoid cells in the source districts, not
buffered positive cell-days. The pairwise ranking loss compares each positive
with unreported, zero-own-history cells in the same district on the same day;
one comparison has a similar nearby-history score. Those comparisons are
unlabeled, **not verified dry**.

| Training arm | Additional training step | Q4 250 m new-site capture, applying its source-selected blend in every fold |
| --- | --- | ---: |
| New-site positives only | Train the residual from genuine first-in-year site-event examples. | **9.29%** |
| New sites + group mask | Add repeat-site examples after hiding the 3 × 3 historical-report neighborhood around the site; weight them 0.3 in the loss. | 7.94% |

For each geographic fold, the history/CNN blend and the choice between these
two arms use **January–September 2025 source-district** new-site capture. The
source choice picks the new-site-only arm in folds 1, 4, and 5, and the group
mask in folds 2 and 3; the combined Q4 result is **10.01%**. The group mask is
not a reliable improvement. Within the 250 m Q4 target, **393 of 471** site
events have zero prior reports in their representative cell and can be
reordered by the CNN; the other 78 use the common known-site history queue.

The raw-site label aligns with the earlier raster target: **2,324 of 2,326**
Q4 site medoid cells are report-positive in the older buffered cell grid. The
two mismatches remain in the raw-site evaluation because the raw coordinates
are the direct outcome source.

## Interpretation and next decision

This is evidence that **cleaner site labels plus training on that target** can
improve the inspection ranking beyond the previous cell-trained model. It does
not validate flood probability, water extent, depth, or rain scenarios. A
larger model alone did not produce this result; the target and comparison
design changed. The 250 m radius was chosen as a plausible site definition
after inspecting the old Q4 errors, and the 100/500 m rows show that the
answer changes materially with that definition. A later untouched season is
needed before treating the 10.01% as an expected operating result. Further
work should concentrate on the **>750 m** sites and on time-matched rainfall
and drainage state rather than more synthetic repeat-site masks.

## Reproduce

Run, in order, with `.venv/bin/python`:

1. [Physical-site construction](flood_citywide_physical_site_data.py)
2. [Frozen old-model replay](flood_citywide_physical_site_replay.py)
3. [Site-residual training](flood_citywide_physical_site_residual.py)
4. [Paired final comparison](flood_citywide_physical_site_final.py)

The [site label audit](artifacts/flood_open_weak_train/multi_target/citywide/report_physical_site_data_audit.json), [old-model replay](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_replay.json), [training result](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_residual_fivefold.json), [final metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_final.json), and [event-level comparison](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_physical_site_final_events_250m.csv) are saved. Raw ticket text, photos, and addresses were not used.
