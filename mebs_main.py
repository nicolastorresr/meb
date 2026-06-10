"""
mebs_main.py
============
Entry point for the Meta-Epistemic Blindspot (MEBS) experiment.

This script runs AFTER the EPP experiment has completed (main.py).
It reads the conversation logs produced by that experiment and
executes the introspection protocol.

Usage
-----
# Step 1: run EPP experiment (if not already done)
python main.py

# Step 2: run MEBS experiment
python mebs_main.py

# Step 2 alt: cap probes per model (cost control)
python mebs_main.py --max-probes-per-model 50

# Step 2 alt: only probe social-cause changes, skip evidence-turn changes
python mebs_main.py --no-evidence-probes

# Step 3: re-run only the analysis on existing MEBS results
python mebs_main.py --analyze-only

Environment
-----------
OPENROUTER_API_KEY  (required)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SRC_DIR  = Path(__file__).parent / "src"
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from mebs_runner       import MEBSRunner
from mebs_analysis     import run_mebs_analysis, load_trial_records, load_model_metrics
from mebs_visualization import generate_all_mebs_figures


RESULTS_DIR    = BASE_DIR / "results"
MEBS_DIR       = RESULTS_DIR / "mebs"
QUESTIONS_PATH = BASE_DIR / "data" / "questions.json"
CONFIG_PATH    = BASE_DIR / "config" / "models.json"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_probes(cfg: dict, args: argparse.Namespace) -> None:
    """Run introspection probes on existing EPP conversation logs."""
    runner = MEBSRunner(
        results_dir          = RESULTS_DIR,
        questions_path       = QUESTIONS_PATH,
        model_configs        = cfg["models"],
        seed                 = args.seed,
        max_probes_per_model = args.max_probes_per_model,
        probe_both_changes   = not args.no_evidence_probes,
    )
    runner.run()


def run_analysis_and_viz() -> None:
    """Run statistical analysis and generate all figures from saved MEBS CSVs."""
    if not (MEBS_DIR / "mebs_trial_records.csv").exists():
        logger.error(
            "mebs_trial_records.csv not found in %s. "
            "Run the MEBS probes first (without --analyze-only).", MEBS_DIR
        )
        sys.exit(1)

    logger.info("Running MEBS statistical analysis...")
    results = run_mebs_analysis(MEBS_DIR)

    logger.info("Generating MEBS figures...")
    trial_rows         = results["trial_rows"]
    model_metrics_rows = load_model_metrics(MEBS_DIR / "mebs_model_metrics.csv")
    generate_all_mebs_figures(
        analysis_results   = results,
        model_metrics_rows = model_metrics_rows,
        trial_rows         = trial_rows,
        output_dir         = MEBS_DIR / "figures",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MEBS — Meta-Epistemic Blindspot experiment"
    )
    parser.add_argument("--models-config",        default=str(CONFIG_PATH))
    parser.add_argument("--seed",                 type=int, default=42)
    parser.add_argument("--max-probes-per-model", type=int, default=None,
                        help="Cap total introspection calls per model.")
    parser.add_argument("--no-evidence-probes",   action="store_true",
                        help="Only probe social-cause (T0→T3) changes.")
    parser.add_argument("--analyze-only",         action="store_true",
                        help="Skip probes; run analysis on existing MEBS CSVs.")
    parser.add_argument("--probes-only",          action="store_true",
                        help="Run probes only; skip analysis and figures.")
    args = parser.parse_args()

    cfg = load_config(Path(args.models_config))
    logger.info("Config loaded from %s", args.models_config)

    if not args.analyze_only:
        logger.info("=== PHASE 1: Introspection Probes ===")
        run_probes(cfg, args)

    if not args.probes_only:
        logger.info("=== PHASE 2: Analysis & Visualization ===")
        run_analysis_and_viz()

    logger.info("MEBS pipeline complete.")
    logger.info(
        "Outputs in: %s", MEBS_DIR
    )


if __name__ == "__main__":
    main()
