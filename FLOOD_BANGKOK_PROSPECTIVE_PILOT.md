# Prospective Bangkok flood-inspection pilot kit

The [retrospective budget test](FLOOD_BANGKOK_INSPECTION_BUDGET.md) cannot establish field savings: it has later complaints but no actual inspection queue, reviewed wet/dry observations, or staff time. This kit defines and checks those records. **No prospective queues or field outcomes have been collected yet.** The included 2025 files are a historical workflow rehearsal.

## Trial design

Use one district and one full rainy season. Bang Khae is the initial study area because its 840 model cells and earlier backtests are available. Before the season, freeze:

1. **25 model cells** ranked from an as-of-date score snapshot.
2. **25 actual current-practice cells**, supplied and timestamped by the district team before seeing future outcomes. Record the reason for each selection. The open-data “recent reports” rule can be used for a demo, but is not BMA practice.
3. **10 random eligible cells** outside both queues to audit overlooked areas.

Inspect the union once per rain event; overlapping model/practice cells count for both policies but require only one physical visit. The historical rehearsal has 13 overlapping cells and 47 unique cells in this union. Field teams should rotate visit order and use the same time window and checklist for all three groups. Do not reveal ranking scores to the photo reviewer. Record any drain clearing or other intervention because it can change later flooding.

Log every district rain-response alert used to trigger dispatch, before knowing inspection findings. The primary finding is a **new distinct cell with standing water at least 10 cm deep**, observed from alert time through three hours after rain ends, with a geotagged, timestamped photo and an independent reviewer. The code accepts an observed position within 250 m of the selected cell centre. A visit after the window or a missing photo is **unknown for the primary finding**, not verified dry. Record drain blockage separately. The primary comparison is distinct verified wet cells per 10 staff-hours; also show verified cell-events, queue coverage, and median delay from alert. Staff-hours include each visit's inbound travel and onsite time multiplied by staff count. The analysis counts one overlapping visit in both policy denominators as the counterfactual work either queue would request.

Treat a larger rate as a reason for broader testing, not proof of citywide savings. Compare the same rain events and report differences in visit timing and completion. The first season may have too few storms or verified sites for a stable effect. Sites fixed after inspection can disappear from later flood reports, so report findings at inspection separately from later complaints.

## Freeze before the season

The [score snapshot template](artifacts/flood_bangkok_field_pilot/score_snapshot_template.csv) needs `cell_id,lon,lat,score_asof,model_score,history_score` for every eligible cell. Rebuild it from reports and rain available by the decision date; do not use later static map updates without recording their dates. The [current-practice template](artifacts/flood_bangkok_field_pilot/current_practice_queue_template.csv) needs the district's actual 25 ranked cell IDs, selection reasons, and timestamps. The freeze command rejects a prospective run with no practice queue or a date after the evaluation window has started. It also rejects outcome columns in the score snapshot.

```sh
.venv/bin/python flood_bangkok_field_pilot.py freeze \
  --scores path/to/asof_score_snapshot.csv \
  --practice path/to/current_practice_queue.csv \
  --decision-date YYYY-MM-DD \
  --evaluation-start YYYY-MM-DD \
  --evaluation-end YYYY-MM-DD \
  --budget 25 --random-control 10 \
  --out artifacts/flood_bangkok_field_pilot/live_season
```

Store the output `freeze_manifest.json` and its queue SHA-256 hash with someone outside the field team before observations begin. A hash kept only beside an editable queue does not independently prove that the queue was frozen. The output `frozen_queue.csv` contains no future outcomes. Keep the model and practice cells unchanged throughout the season; document any cells that cannot be visited.

## Record and analyse visits

Copy the frozen directory's blank `rain_events_template.csv` and `field_visits_template.csv`. Use ISO timestamps with an explicit `+07:00` offset. Give each triggered rain episode a unique `event_id`. For each visit record cell ID, inspector, timestamps, staff count, inbound travel minutes, onsite minutes, observed wet/dry/unknown status, depth, drain defect, geotag, photo reference, separate reviewer, and any intervention. A photo should be retained in a controlled location and referenced by ID rather than copied into the public CSV. Reviewer status is `confirmed`, `rejected`, or `pending`.

```sh
.venv/bin/python flood_bangkok_field_pilot.py analyze \
  --queue artifacts/flood_bangkok_field_pilot/live_season/frozen_queue.csv \
  --events path/to/rain_events.csv \
  --visits path/to/field_visits.csv \
  --out artifacts/flood_bangkok_field_pilot/live_season/field_results.json
```

The analyzer reports provisional status while any review remains pending. It does not invent findings for unvisited cells. Before interpreting policy rates, check queue coverage and visit timing by arm; a queue visited earlier in a storm could appear better simply because water later receded.

## Historical workflow rehearsal

`.venv/bin/python flood_bangkok_field_pilot.py demo` strips future-outcome columns from the 2025 Bang Khae ranked-cell artifact and creates a [demo score snapshot](artifacts/flood_bangkok_field_pilot/historical_demo_scores_2025.csv), [queue](artifacts/flood_bangkok_field_pilot/historical_demo_2025/frozen_queue.csv), [blank field sheet](artifacts/flood_bangkok_field_pilot/historical_demo_2025/field_visits_template.csv), [blank event log](artifacts/flood_bangkok_field_pilot/historical_demo_2025/rain_events_template.csv), and [manifest](artifacts/flood_bangkok_field_pilot/historical_demo_2025/freeze_manifest.json). Its practice arm is the recent-report heuristic. Its decision date was in **2025**, but the manifest was created later; it is **not** a prospective trial or evidence of verified findings. Fresh as-of-date score inputs and an actual district queue are required for the real freeze.
