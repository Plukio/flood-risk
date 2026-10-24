"""Check weak flood-label scenes before training or interpreting their scores.

Run: .venv/bin/python flood_label_audit.py

The wet array is mapped proxy water. For UNOSAT, observed is the analyzed
footprint after exclusions and can support *weak* nonwater comparisons. For
EOS-RS and GISTDA, observed must be zero: transparency is unknown ground.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "artifacts/flood_open_weak_train/data_audit.json"
OUTPUT = ROOT / "artifacts/flood_open_weak_train/label_readiness_audit.json"
SOURCE_STATUS = {
    "unosat2021": "analysis_extent_weak_nonwater",
    "unosat2024": "analysis_extent_weak_nonwater",
    "eos2025": "positive_only",
    "eos2026": "positive_only",
    "gistda2026": "positive_only",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_path(entry: dict) -> Path:
    relative = Path(entry["path"])
    if relative.is_absolute() or relative.suffix != ".npz":
        raise ValueError(f"Scene path must be a relative .npz: {relative}")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Scene path missing or outside project: {relative}")
    return path


def validate_scene(entry: dict, arrays: dict, manifest: dict) -> dict:
    """Return measured label coverage; reject accidental dry labels or bad data."""
    source = entry["source"]
    if source not in SOURCE_STATUS:
        raise ValueError(f"Unknown label source: {source}")
    date.fromisoformat(entry["day"])
    h, w = manifest["patch_size"]
    expected = {"wet": (h, w), "observed": (h, w),
                "static": (len(manifest["features"]), h, w),
                "weather": (len(manifest["weather"]),)}
    for name, shape in expected.items():
        value = arrays[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{entry['path']}: invalid {name} shape or finite values")
    wet = arrays["wet"]
    observed = arrays["observed"]
    if np.any((wet < -1e-6) | (wet > 1 + 1e-6)) or np.any(
            (observed < -1e-6) | (observed > 1 + 1e-6)):
        raise ValueError(f"{entry['path']}: label fractions must be in [0, 1]")
    if SOURCE_STATUS[source] == "positive_only":
        if np.any(observed > 1e-6):
            raise ValueError(f"{entry['path']}: positive-only transparency was marked observed")
    elif np.any(wet > observed + 1e-6):
        raise ValueError(f"{entry['path']}: mapped water lies outside analysis extent")
    if not np.any(wet > 0) and not np.any(observed > wet):
        raise ValueError(f"{entry['path']}: scene has no usable label")

    area_km2 = (float(manifest["cell_m"]) / 1000) ** 2
    wet_area = float(np.sum(wet, dtype=np.float64) * area_km2)
    observed_area = float(np.sum(observed, dtype=np.float64) * area_km2)
    for field, measured in (("wet_proxy_km2", wet_area),
                            ("observed_candidate_km2", observed_area)):
        if not math.isclose(float(entry[field]), measured, abs_tol=0.00011):
            raise ValueError(f"{entry['path']}: {field} disagrees with array")
    return {"wet_proxy_km2": round(wet_area, 4),
            "observed_candidate_km2": round(observed_area, 4),
            "weak_nonwater_km2": round(max(0., observed_area - wet_area), 4)
            if SOURCE_STATUS[source] != "positive_only" else 0.,
            "label_status": SOURCE_STATUS[source]}


def audit(manifest_path: Path = MANIFEST, output_path: Path = OUTPUT) -> dict:
    manifest = json.loads(manifest_path.read_text())
    seen = set()
    summaries = defaultdict(lambda: {"scenes": 0, "wet_proxy_km2_days": 0.,
                                     "observed_candidate_km2_days": 0.,
                                     "weak_nonwater_km2_days": 0.})
    rows = []
    for entry in manifest["scenes"]:
        path = scene_path(entry)
        if path in seen:
            raise ValueError(f"Duplicate scene path: {path}")
        seen.add(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in
                      ("wet", "observed", "static", "weather")}
        measured = validate_scene(entry, arrays, manifest)
        source = entry["source"]
        summary = summaries[source]
        summary["scenes"] += 1
        for key in ("wet_proxy_km2", "observed_candidate_km2",
                    "weak_nonwater_km2"):
            summary[key + "_days"] += measured[key]
        rows.append({"source": source, "region": entry["region"],
                     "day": entry["day"], "path": entry["path"],
                     "sha256": sha256(path), **measured,
                     "acquisition_at": None,
                     "pre_observation_rain_verified": False,
                     "verified_urban_wet_dry": False})
    if not rows:
        raise ValueError("Manifest contains no scenes")
    for values in summaries.values():
        for key in ("wet_proxy_km2_days", "observed_candidate_km2_days",
                    "weak_nonwater_km2_days"):
            values[key] = round(values[key], 4)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "scene_count": len(rows), "by_source": dict(sorted(summaries.items())),
        "time_assessment": "Scene manifest records product/event day, not a verified acquisition hour; weather is retrospective full-day exposure.",
        "use_gate": {
            "positive_only_proxy_ranking": True,
            "weak_nonwater_comparison": "UNOSAT analysis extent only, with radar urban misses",
            "pre_event_rain_forecasting": False,
            "calibrated_urban_flood_probability": False,
            "flood_depth_or_counterfactual_simulation": False,
        },
        "scenes": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = audit()
    print(f"Validated {result['scene_count']} weak-label scenes; wrote {OUTPUT}")
    for source, values in result["by_source"].items():
        print(source, values)
