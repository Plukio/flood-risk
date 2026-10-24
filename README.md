# Bangkok flood-risk triage

Research code, audit notes, and a working paper for prioritizing Bangkok 250 m map squares for flood-site review. The ranking is an exploratory decision aid, not a flood-depth or avoided-damage forecast.

## Current result

On the Q4 2025 held-out district evaluation, screening the top 5% of **unfamiliar cells** captured 13.96% of 393 later new report-site events with the proposed spatial model, versus 6.82% with report history. When every district cell competed for the same top-5% shortlist, both methods captured 1.27% of 471 emerging events. The economic examples in the paper are explicitly hypothetical.

The main paper is [BANGKOK_FLOOD_SITE_RISK_PAPER.pdf](BANGKOK_FLOOD_SITE_RISK_PAPER.pdf). The [current-state summary](FLOOD_CURRENT_STATE.md), [site-ranking audit](FLOOD_SITE_QUOTA_FREE_RANKING.md), and [full-cell queue audit](FLOOD_SITE_ALL_CELL_CURVES.md) explain the evaluation and its limits.

## Rebuild the paper

The small, committed `artifacts/flood_open_weak_train/multi_target/citywide/` files contain the rank curves, site-event rows, and building-footprint overlay read by the paper generator.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-paper.txt
.venv/bin/python build_flood_site_academic_paper.py
```

This reproduces the paper from saved study outputs without downloading external data. The builder uses Python 3.12, NumPy 2.5.4, and Matplotlib 3.11.2. The committed PDF is the rendered output.

## Re-running the analyses

The Python scripts and research notes remain in this repository, including the report-history benchmark, spatial model, site-label construction, and held-out evaluation. Full retraining and regeneration of every historical result require external source datasets and intermediate arrays that are **not** committed: Traffy reports, BMA flood and drainage records, OpenStreetMap extracts, terrain and land-cover data, and selected rainfall and satellite products. See [open-data audit](FLOOD_OPEN_DATA_AUDIT_2026.md) and the paper references for sources and dates. The `data/` directory, model checkpoints, large generated artifacts, logs, and local previews are excluded from Git.

The committed building-footprint overlay derives from an OpenStreetMap snapshot dated 1 January 2025. © OpenStreetMap contributors; Open Database License. Footprint area is only a rough exposure proxy: it does not measure floor area, businesses, traffic, or economic loss.

## Reproducibility boundary

The committed inputs reproduce the **PDF and its reported calculations**. They do not by themselves reproduce model training or establish verified flood extent, prevented losses, or an operational gain at the top 5% of all district cells.
