"""Freeze a Bangkok flood-inspection pilot and analyse reviewed field visits.

The included 2025 run is a historical workflow rehearsal, not a prospective
field result. A real freeze requires a dated score snapshot and an actual
current-practice queue, both supplied before the evaluation window.

Run: .venv/bin/python flood_bangkok_field_pilot.py demo
     .venv/bin/python flood_bangkok_field_pilot.py freeze --help
     .venv/bin/python flood_bangkok_field_pilot.py analyze --help
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "flood_bangkok_field_pilot"
HISTORICAL = ROOT / "artifacts" / "flood_bangkapi_surface" / \
    "surface_inspection_budget_backtest_cell_ranks.csv"
SCORE_COLUMNS = ("cell_id", "lon", "lat", "score_asof", "model_score",
                 "history_score")
VISIT_COLUMNS = ("event_id", "cell_id", "inspector_id",
                 "visit_started_at", "visit_ended_at",
                 "staff_count", "travel_minutes", "onsite_minutes",
                 "wet_status", "depth_cm", "drainage_defect", "photo_ref",
                 "observed_lon", "observed_lat",
                 "review_status", "reviewer_id", "intervention_type",
                 "intervention_at", "notes")
EVENT_COLUMNS = ("event_id", "alert_at", "rain_end_at", "trigger_source",
                 "rain_mm", "notes")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(rows: list[dict], names: tuple[str, ...], path: Path) -> None:
    if not rows or not set(names).issubset(rows[0]):
        raise ValueError(f"{path} needs columns {names} and at least one row")


def create_demo_scores() -> Path:
    source = read_csv(HISTORICAL)
    source = [row for row in source if row["district_code"] == "1040"
              and row["year"] == "2025"]
    if len(source) != 840:
        raise ValueError("Historical Bang Khae ranking is incomplete")
    safe = [{"cell_id": row["cell_id"], "lon": row["lon"],
             "lat": row["lat"], "score_asof": "2025-09-30",
             "model_score": row["local_baseline_score"],
             "history_score": row["recent_report_days_score"]}
            for row in source]
    path = OUT / "historical_demo_scores_2025.csv"
    write_csv(path, safe, SCORE_COLUMNS)
    return path


def freeze(score_path: Path, practice_path: Path | None, output_dir: Path,
           decision_day: date, eval_start: date, eval_end: date,
           budget: int, controls: int, seed: int, demo: bool) -> dict:
    if not decision_day < eval_start <= eval_end:
        raise ValueError("Decision date must precede evaluation dates")
    if not demo and decision_day > datetime.now(timezone.utc).date():
        raise ValueError("Cannot freeze a score snapshot dated in the future")
    if not demo and eval_start <= datetime.now(timezone.utc).date():
        raise ValueError("Prospective freeze must happen before evaluation starts")
    if not demo and practice_path is None:
        raise ValueError("A real pilot requires an actual current-practice queue")
    scores = read_csv(score_path)
    require_columns(scores, SCORE_COLUMNS, score_path)
    if len({row["cell_id"] for row in scores}) != len(scores):
        raise ValueError("Duplicate cells in score snapshot")
    if any(row["score_asof"] != str(decision_day) for row in scores):
        raise ValueError("Every score row must carry the frozen decision date")
    if any(word in name.lower() for name in scores[0]
           for word in ("future", "outcome", "observed", "test_")):
        raise ValueError("Score snapshot contains future-outcome columns")
    if not 0 < budget < len(scores) or not 0 <= controls <= len(scores) - 2 * budget:
        raise ValueError("Invalid budget or random-control count")
    by_id = {row["cell_id"]: row for row in scores}
    for row in scores:
        lon, lat = float(row["lon"]), float(row["lat"])
        if not (100 < lon < 101 and 13 < lat < 14):
            raise ValueError(f"Cell is outside the Bangkok coordinate range: {row['cell_id']}")
        if not (math.isfinite(float(row["model_score"])) and
                math.isfinite(float(row["history_score"]))):
            raise ValueError(f"Nonfinite score: {row['cell_id']}")
    model_order = sorted(by_id, key=lambda cid: (-float(by_id[cid]["model_score"]), cid))
    if practice_path is None:
        if not demo:
            raise ValueError("Current-practice queue missing")
        practice_method = "recent_report_days_heuristic_demo"
        practice_order = sorted(by_id, key=lambda cid: (-float(by_id[cid]["history_score"]), cid))
    else:
        practice = read_csv(practice_path)
        require_columns(practice, ("cell_id", "practice_rank", "selected_at",
                                   "selection_reason"), practice_path)
        if len(practice) != budget or len({r["cell_id"] for r in practice}) != budget:
            raise ValueError("Current-practice file must list exactly one budget of unique cells")
        if any(r["cell_id"] not in by_id for r in practice):
            raise ValueError("Current-practice cell is absent from the score grid")
        ranks = [int(r["practice_rank"]) for r in practice]
        if sorted(ranks) != list(range(1, budget + 1)):
            raise ValueError("Current-practice ranks must be 1..budget")
        for row in practice:
            chosen = datetime.fromisoformat(row["selected_at"])
            if (chosen.tzinfo is None or chosen.date() > decision_day
                    or not row["selection_reason"].strip()):
                raise ValueError("Practice selections need dated reasons before the decision cutoff")
        practice_order = [r["cell_id"] for r in sorted(practice,
                          key=lambda row: int(row["practice_rank"]))]
        practice_method = "actual_current_practice_supplied"
    selected_model = set(model_order[:budget])
    selected_practice = set(practice_order[:budget])
    remaining = sorted(set(by_id) - selected_model - selected_practice)
    selected_control = set(random.Random(seed).sample(remaining, controls))
    union = selected_model | selected_practice | selected_control
    queue = []
    for cid in sorted(union):
        row = by_id[cid]
        queue.append({"cell_id": cid, "lon": row["lon"], "lat": row["lat"],
                      "model_selected": int(cid in selected_model),
                      "practice_selected": int(cid in selected_practice),
                      "random_control": int(cid in selected_control),
                      "model_rank": model_order.index(cid) + 1 if cid in selected_model else "",
                      "practice_rank": practice_order.index(cid) + 1
                      if cid in selected_practice else ""})
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "frozen_queue.csv"
    write_csv(queue_path, queue, ("cell_id", "lon", "lat", "model_selected",
                                  "practice_selected", "random_control",
                                  "model_rank", "practice_rank"))
    # A blank row is only a schema template. Visits are entered after actual
    # field work; no synthetic wet, dry, or time observations are created.
    visit_path = output_dir / "field_visits_template.csv"
    write_csv(visit_path, [], VISIT_COLUMNS)
    write_csv(output_dir / "rain_events_template.csv", [], EVENT_COLUMNS)
    manifest = {
        "status": "historical_workflow_demo" if demo else "prospective_queue_frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_date": str(decision_day),
        "evaluation_start": str(eval_start), "evaluation_end": str(eval_end),
        "budget_per_policy": budget, "random_control_cells": controls,
        "random_seed": seed, "practice_method": practice_method,
        "eligible_cells": len(scores), "model_cells": len(selected_model),
        "practice_cells": len(selected_practice),
        "queue_overlap_cells": len(selected_model & selected_practice),
        "physical_cells_in_union": len(union),
        "score_file": str(score_path.resolve()), "score_sha256": sha256(score_path),
        "practice_file": str(practice_path.resolve()) if practice_path else None,
        "practice_sha256": sha256(practice_path) if practice_path else None,
        "queue_sha256": sha256(queue_path),
        "analysis_rule": "Distinct verified wet cells per staff-hour is primary; same visit credited to both queues if selected by both; unvisited and pending observations reported separately",
        "warning": "This local manifest detects changes only if its hash is independently retained. A historical demo is not a prospective field result.",
    }
    (output_dir / "freeze_manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def analyze(queue_path: Path, events_path: Path, visits_path: Path,
            output_path: Path) -> dict:
    queue = read_csv(queue_path)
    events = read_csv(events_path)
    visits = read_csv(visits_path)
    require_columns(queue, ("cell_id", "model_selected", "practice_selected",
                            "random_control"), queue_path)
    require_columns(visits, VISIT_COLUMNS, visits_path)
    require_columns(events, EVENT_COLUMNS, events_path)
    by_id = {row["cell_id"]: row for row in queue}
    if len(by_id) != len(queue):
        raise ValueError("Duplicate queue cell")
    by_event = {}
    for event in events:
        event_id = event["event_id"]
        if not event_id or event_id in by_event:
            raise ValueError("Missing or duplicate rain event ID")
        alert = datetime.fromisoformat(event["alert_at"])
        rain_end = datetime.fromisoformat(event["rain_end_at"])
        if alert.tzinfo is None or rain_end.tzinfo is None or rain_end < alert:
            raise ValueError("Event times need time zones and rain_end after alert")
        by_event[event_id] = (alert, rain_end)
    arm_minutes = defaultdict(float)
    arm_onsite_minutes = defaultdict(float)
    arm_verified = defaultdict(set)
    arm_visited = defaultdict(set)
    arm_delays = defaultdict(list)
    arm_in_window = defaultdict(int)
    review_pending = 0
    event_ids = set()
    for row in visits:
        cid = row["cell_id"]
        if cid not in by_id or row["event_id"] not in by_event:
            raise ValueError(f"Visit has unknown cell or missing event: {cid}")
        if not row["inspector_id"]:
            raise ValueError("Every visit needs a pseudonymous inspector ID")
        start = datetime.fromisoformat(row["visit_started_at"])
        end = datetime.fromisoformat(row["visit_ended_at"])
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            raise ValueError("Visit timestamps must include time zone and end after start")
        staff = int(row["staff_count"])
        travel, onsite = float(row["travel_minutes"]), float(row["onsite_minutes"])
        if staff < 1 or travel < 0 or onsite <= 0:
            raise ValueError("Invalid staff count or effort minutes")
        if abs((end - start).total_seconds() / 60 - onsite) > 15:
            raise ValueError("Onsite minutes disagree with visit timestamps by >15 min")
        alert, rain_end = by_event[row["event_id"]]
        window_ok = alert <= start <= rain_end + timedelta(hours=3)
        if row["wet_status"] not in ("wet", "dry", "unknown"):
            raise ValueError("wet_status must be wet, dry, or unknown")
        if row["review_status"] not in ("confirmed", "rejected", "pending"):
            raise ValueError("review_status must be confirmed, rejected, or pending")
        if row["review_status"] == "confirmed" and (
                not row["reviewer_id"] or row["reviewer_id"] == row["inspector_id"]):
            raise ValueError("A confirmed visit needs a different reviewer ID")
        if row["review_status"] == "pending":
            review_pending += 1
        depth = float(row["depth_cm"]) if row["depth_cm"] else None
        if depth is not None and depth < 0:
            raise ValueError("Negative water depth")
        confirmed_wet = (row["review_status"] == "confirmed"
                         and window_ok
                         and row["wet_status"] == "wet"
                         and depth is not None and depth >= 10
                         and bool(row["photo_ref"])
                         and bool(row["reviewer_id"])
                         and row["reviewer_id"] != row["inspector_id"])
        if confirmed_wet:
            if not row["observed_lon"] or not row["observed_lat"]:
                raise ValueError("Confirmed wet visit needs observed GPS coordinates")
            lon, lat = float(row["observed_lon"]), float(row["observed_lat"])
            centre_lon, centre_lat = float(by_id[cid]["lon"]), float(by_id[cid]["lat"])
            dx = (lon - centre_lon) * 111_000 * math.cos(math.radians(centre_lat))
            dy = (lat - centre_lat) * 111_000
            if math.hypot(dx, dy) > 250:
                raise ValueError("Confirmed wet observation is >250 m from selected cell centre")
        event_ids.add(row["event_id"])
        for flag, arm in (("model_selected", "model"),
                          ("practice_selected", "practice"),
                          ("random_control", "random_control")):
            if by_id[cid][flag] != "1":
                continue
            arm_minutes[arm] += staff * (travel + onsite)
            arm_onsite_minutes[arm] += staff * onsite
            arm_visited[arm].add(cid)
            arm_delays[arm].append((start - alert).total_seconds() / 60)
            arm_in_window[arm] += int(window_ok)
            if confirmed_wet:
                arm_verified[arm].add((row["event_id"], cid))
    result = {"status": "provisional_pending_review" if review_pending else "reviewed",
              "logged_rain_events": len(by_event), "events_with_visits": len(event_ids),
              "visits": len(visits),
              "pending_reviews": review_pending, "arms": {}}
    for arm, flag in (("model", "model_selected"),
                      ("practice", "practice_selected"),
                      ("random_control", "random_control")):
        selected = sum(row[flag] == "1" for row in queue)
        event_cells = arm_verified[arm]
        hours = arm_minutes[arm] / 60
        unique_sites = len({cid for _, cid in event_cells})
        result["arms"][arm] = {
            "selected_cells": selected, "visited_cells": len(arm_visited[arm]),
            "visits_within_alert_to_3h_post_rain_window": arm_in_window[arm],
            "median_visit_delay_from_alert_minutes": round(
                statistics.median(arm_delays[arm]), 1) if arm_delays[arm] else None,
            "staff_hours": round(hours, 3),
            "onsite_staff_hours": round(arm_onsite_minutes[arm] / 60, 3),
            "verified_wet_cell_events": len(event_cells),
            "distinct_verified_wet_cells": unique_sites,
            "distinct_wet_cells_per_10_staff_hours": round(
                10 * unique_sites / hours, 3) if hours else None,
            "verified_wet_events_per_10_staff_hours": round(
                10 * len(event_cells) / hours, 3) if hours else None}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--scores", type=Path, required=True)
    freeze_parser.add_argument("--practice", type=Path, required=True)
    freeze_parser.add_argument("--decision-date", type=date.fromisoformat, required=True)
    freeze_parser.add_argument("--evaluation-start", type=date.fromisoformat, required=True)
    freeze_parser.add_argument("--evaluation-end", type=date.fromisoformat, required=True)
    freeze_parser.add_argument("--budget", type=int, default=25)
    freeze_parser.add_argument("--random-control", type=int, default=10)
    freeze_parser.add_argument("--seed", type=int, default=7)
    freeze_parser.add_argument("--out", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--queue", type=Path, required=True)
    analyze_parser.add_argument("--events", type=Path, required=True)
    analyze_parser.add_argument("--visits", type=Path, required=True)
    analyze_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        scores = create_demo_scores()
        write_csv(OUT / "current_practice_queue_template.csv",
                  [{"practice_rank": i, "cell_id": "", "selected_at": "",
                    "selection_reason": ""} for i in range(1, 26)],
                  ("practice_rank", "cell_id", "selected_at", "selection_reason"))
        write_csv(OUT / "score_snapshot_template.csv", [], SCORE_COLUMNS)
        result = freeze(scores, None, OUT / "historical_demo_2025",
                        date(2025, 9, 30), date(2025, 10, 1),
                        date(2025, 12, 31), 25, 10, 7, True)
    elif args.command == "freeze":
        result = freeze(args.scores, args.practice, args.out,
                        args.decision_date, args.evaluation_start,
                        args.evaluation_end, args.budget,
                        args.random_control, args.seed, False)
    else:
        result = analyze(args.queue, args.events, args.visits, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
