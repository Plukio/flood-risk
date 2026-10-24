# Bang Kapi road-location ranking: open-data backtest

Run `.venv/bin/python flood_bangkapi_site_ranking.py` from this directory. The
script writes an [all-report match audit](artifacts/flood_bangkapi_site_ranking/report_match_audit.csv),
[full-year site ranking](artifacts/flood_bangkapi_site_ranking/site_ranking_2025.csv),
[forward ranking](artifacts/flood_bangkapi_site_ranking/known_site_forward_ranking_2025.csv),
and [metrics](artifacts/flood_bangkapi_site_ranking/metrics.json).

## What was ranked

The unit is a **named road location**, such as Ramkhamhaeng Soi 33, not a
10–50 m OpenStreetMap road piece. BMA recorded 44 Bang Kapi major-road flood
report rows in 2025. All 44 can be matched by exact or explicit place aliases
to the point inventory; two repeat the same site on the same date, leaving
**42 site report-days on 12 dates**. The 14 inventory points become 13 named
locations because two have the same Soi 33 name. BMA text cannot tell which
of those two points flooded. Of eight Bang Kapi rows from 2021–2024, four
match; four describe broad or ambiguous corridors and remain unmatched.

The location coordinates come from [BMA's 2026 flood-risk-point inventory](https://data.bangkok.go.th/dataset/flood_risk26).
This inventory was published **after** the 2025 outcome year. It is useful for
mapping names to terrain, but its selection may reflect the same 2025 floods.
The full-year 13-site terrain check is therefore an exploratory retrospective
comparison, not a clean prospective test.

## Direct rank results

LiDAR terrain uses a 1 m [BMA DTM export](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson)
around each inventory point. The site elevation is the median within 20 m;
local depression compares that median with a 40–110 m surrounding ring.
Scores were given their physical directions before comparison: lower terrain
and deeper relative depression rank as higher risk. The historical baseline
uses only 2021–2024 BMA report days.

| Full-year 2025 ranking, 13 named sites | Spearman against 2025 report-day counts | Correct order of 64 comparable pairs |
|---|---:|---:|
| Low absolute elevation | −0.159 | 43.8% |
| Local terrain depression | −0.524 | 28.1% |
| 2021–2024 report days | −0.080 | 48.4% |

The terrain scores fail this site ranking. For example, Hua Mak / Ramkhamhaeng
24 intersection 18 has **10** report days but ranks **12th of 13** by local
depression. The motorway underpass ranks first by both terrain measures but
has one report day. This does not prove terrain is unrelated to flood depth:
the motorway point is **36.8 m from the nearest OSM drivable road centreline**,
outside the 20 m LiDAR sample core. The other 12 points are within 9 m of an
OSM centreline, though the nearest way need not be the road named in a report.
Road drainage and pumping are missing, and
report frequency reflects which roads BMA monitors and records.

## Forward hotspot check

To avoid selecting sites using future reports, the forward check forms its
cohort solely from locations reported in **May–August 2025**. Ten locations
qualify. Their May–August report-day counts rank their **September–December
2025** report-day counts. The 2026 coordinates are not used in this ranking.

| Known location | May–Aug report days | Sep–Dec report days |
|---|---:|---:|
| Hua Mak / Ramkhamhaeng 24 intersection 18 | 5 | 5 |
| Ramkhamhaeng Soi 33 | 3 | 4 |
| Green Sport | 2 | 0 |
| Ramkhamhaeng Soi 19/1 | 2 | 2 |
| Ramkhamhaeng Soi 20 | 2 | 2 |
| Bang Kapi intersection | 1 | 2 |
| Lam Sali intersection | 1 | 2 |
| Opposite Lat Phrao police station | 1 | 0 |
| Ramkhamhaeng Soi 43 opposite | 1 | 3 |
| Ramkhamhaeng Soi 53 opposite | 1 | 0 |

**Spearman = 0.528**, with 72.2% correct order across 36 comparable pairs
(unadjusted Spearman *p* = 0.117; ten sites). These ten locations account for
20 of 23 September–December site report-days. The other three are at places
not reported before September: the motorway underpass, Lat Phrao Soi 110–112,
and the Ramkhamhaeng 24 U-turn. This is a practical **repeat-report hotspot
ranking**, not a flood probability or validated street-flood model. The cohort
is small and based on retrospective place-name matching. The 2025 BMA report
file was itself published later, so this is a chronological backtest; it does
not establish that the same report feed was publicly available in real time.

## Implication for the hackathon

The useful open-data prototype is a map of recent reported hotspots, with
terrain and BMA drainage assets as explanatory layers and a queue for **new**
reports. Rainfall amount and duration can drive district storm scenarios, but
the present Bang Kapi labels do not establish how each street's risk changes
if rainfall doubles or lasts longer. A U-Net would need event-aligned wet-road
images; [Sentinel-1 radar did not identify a BMA-reported flooded Bang Kapi
road in the pilot](FLOOD_ROAD_DATA_PILOT.md). A separate
[2025 satellite scene-timing audit](FLOOD_BANGKAPI_IMAGERY.md) found only one
Sentinel-1 pass during a reported Bang Kapi flood and no Sentinel-2 pass during
one. The public CCTV catalog provides
locations, not archived frames. The nearby HII BKK008 hourly gauge had a
suspect all-zero October 2025 month, so it was not used to claim a precise
local rainfall response.

For a transferable scenario model, the next data gate is time-stamped wet and
dry observations on consistently geocoded road sections, together with local
rainfall and drainage or pump state. Hold out whole storms and districts and
compare against this simple recent-report baseline. Until then, label map
scores as **report-priority signals**, never as predicted water depth or a
verified road flood.

## Sources

- [BMA major-road flood incidents, 2021–2025](https://data.bangkok.go.th/dataset/frd_dds): road, location text, event date and reported flood depth; 2025 resource was published after the event year.
- [BMA 2026 flood-risk-point inventory](https://data.bangkok.go.th/dataset/flood_risk26): point names and coordinates; post-outcome site-selection caveat above.
- [BMA 2025 flood-risk inventory](https://data.bangkok.go.th/dataset/flood_risk): published in 2025 and inspected as an earlier reference. Its Bang Kapi main-road entries are broad, partly overlapping corridors, so they cannot resolve all 13 named report locations or give unambiguous site labels.
- [BMA LiDAR DTM service](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson): exported 256 m tiles at 1 m resolution around each point.
- [OpenStreetMap Thailand extract](https://download.geofabrik.de/asia/thailand.html): local PBF dated 1 January 2025, used only to audit inventory-point distance to a drivable-road centreline; OpenStreetMap contributors, ODbL.
