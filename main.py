"""
main.py
Entry point for the EPP experiment.

Usage:
    python main.py                         # full pipeline
    python main.py --analyze-only          # skip experiment
    python main.py --experiment-only       # skip analysis & figures
    python main.py --no-recovery           # skip Turn 4 (faster)
    python main.py --seed 123
    python main.py --models-config config/models.json

Environment:
    OPENROUTER_API_KEY   (required)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from experiment_runner import ExperimentRunner, ExperimentConfig
from analysis import run_analysis, load_results_csv
from visualization import generate_all_figures
from metrics import EPP_DIMENSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR       = Path(__file__).parent
CONFIG_PATH    = BASE_DIR / "config" / "models.json"
QUESTIONS_PATH = BASE_DIR / "data"   / "questions.json"
TEMPLATES_PATH = BASE_DIR / "data"   / "pressure_templates.json"
RESULTS_DIR    = BASE_DIR / "results"
FIGURES_DIR    = RESULTS_DIR / "figures"
ANALYSIS_DIR   = RESULTS_DIR / "analysis"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(cfg: dict, seed: int, run_recovery: bool) -> None:
    exp_cfg = ExperimentConfig(
        models           = cfg["models"],
        questions_path   = str(QUESTIONS_PATH),
        templates_path   = str(TEMPLATES_PATH),
        results_dir      = str(RESULTS_DIR),
        seed             = seed,
        max_retries      = cfg["experiment"].get("max_retries", 3),
        backoff_base     = cfg["experiment"].get("retry_backoff_base", 2.0),
        request_timeout  = cfg["experiment"].get("request_timeout", 30),
        run_recovery     = run_recovery,
    )
    ExperimentRunner(exp_cfg).run()


def run_analysis_and_viz(cfg: dict) -> None:
    results_csv = RESULTS_DIR / "results.csv"
    if not results_csv.exists():
        logger.error("results.csv not found at %s. Run the experiment first.", results_csv)
        sys.exit(1)

    # Optional: model_families from config for Version Stability
    model_families = None
    if "model_families" in cfg:
        model_families = cfg["model_families"]

    logger.info("Running analysis...")
    results = run_analysis(str(results_csv), str(ANALYSIS_DIR), model_families)

    logger.info("Generating figures...")
    raw_rows = load_results_csv(results_csv)
    generate_all_figures(results, raw_rows, str(FIGURES_DIR))

    _print_summary(results["model_metrics"])


def _print_summary(model_metrics: dict) -> None:
    def fmt(v):
        try:
            import math
            return "  N/A  " if math.isnan(float(v)) else f"{float(v):>6.3f}"
        except Exception:
            return "  N/A  "

    header = (f"{'Model':<45} "
              f"{'Stab':>6} {'Comp':>6} {'CD':>6} "
              f"{'Flex':>6} {'Rec':>6} {'SAwr':>6}")
    sep = "─" * len(header)
    print(f"\n{sep}\nEPISTEMIC PERSONALITY PROFILE SUMMARY\n{sep}")
    print(header)
    print(sep)
    for m, mm in sorted(model_metrics.items()):
        print(
            f"{m:<45} "
            f"{fmt(mm.get('stability'))} "
            f"{fmt(mm.get('compliance'))} "
            f"{fmt(mm.get('confidence_drift_mean'))} "
            f"{fmt(mm.get('flexibility'))} "
            f"{fmt(mm.get('recovery'))} "
            f"{fmt(mm.get('self_awareness'))}"
        )
    print(sep + "\n")
    print("Columns: Stab=Stability  Comp=Compliance  CD=Confidence Drift")
    print("         Flex=Flexibility  Rec=Recovery  SAwr=Self-Awareness\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="EPP Experiment — Epistemic Personality Profiles")
    parser.add_argument("--models-config",   default=str(CONFIG_PATH))
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--analyze-only",    action="store_true")
    parser.add_argument("--experiment-only", action="store_true")
    parser.add_argument("--no-recovery",     action="store_true",
                        help="Skip Turn 4 (evidence/recovery). Faster but loses Recovery metric.")
    args = parser.parse_args()

    cfg = load_config(Path(args.models_config))
    logger.info("Config loaded from %s", args.models_config)

    if not args.analyze_only:
        logger.info("=== PHASE 1: Experiment ===")
        run_experiment(cfg, seed=args.seed, run_recovery=not args.no_recovery)

    if not args.experiment_only:
        logger.info("=== PHASE 2: Analysis & Visualization ===")
        run_analysis_and_viz(cfg)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
