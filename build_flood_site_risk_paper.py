"""Compatibility entry point for the academic-style Bangkok flood paper.

Run: .venv/bin/python build_flood_site_risk_paper.py
"""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).with_name("build_flood_site_academic_paper.py")),
               run_name="__main__")
