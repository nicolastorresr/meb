"""
mebs_runner.py
==============
Orchestrates the Meta-Epistemic Blindspot (MEBS) experiment.

Can operate in two modes:
  1. ONLINE  — reads a running experiment's conversation logs and fires
               introspection probes in real time (alongside EPP experiment).
  2. OFFLINE — reads an already-completed results.csv + conversation_logs.json
               and runs introspection probes after the fact.

The offline mode is the default because it decouples the two experiments:
  • Run the EPP experiment first (main.py)
  • Then run:  python mebs_runner.py --results-dir results/

Output files (written to results/mebs/):
  mebs_introspections.csv     — one row per introspection probe (Turn 5)
  mebs_trial_records.csv      — one row per trial (with MEBS-relevant fields)
  mebs_model_metrics.csv      — aggregated MEBS metrics per model
  mebs_logs.json              — full introspection conversation logs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── make src/ importable when run directly ───────────────────────────────────
_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from blindspot import (
    IntrospectionResult,
    MEBSMetrics,
    MEBSTrialRecord,
    CauseLabel,
    compute_mebs_metrics,
    run_introspection,
)
from evaluation import evaluate_answer, answer_matches_false_claim


# ── CSV field specs ──────────────────────────────────────────────────────────

INTRO_FIELDNAMES = [
    "experiment_id", "model", "question_id", "question_text",
    "answer_t0", "answer_t3",
    "true_cause", "attribution", "attribution_conf", "explanation",
    "raw_response", "timestamp", "parse_ok",
]

TRIAL_FIELDNAMES = [
    "experiment_id", "model", "question_id", "category", "pressure_type",
    "answer_changed", "answer_t0", "answer_t3",
    "true_cause",
    "attribution", "attribution_conf", "explanation",
    "social_coherence_t3", "epistemic_coherence_t3",
]

METRICS_FIELDNAMES = [
    "model", "n_trials", "n_changed",
    "mebs_score", "attribution_accuracy",
    "social_coherence", "epistemic_coherence", "sc_ec_gap",
    "blindspot_specificity",
    "p_epistemic_given_social", "p_epistemic_given_epistemic",
    "p_social_given_social", "p_social_given_epistemic",
]


# ── data loading ─────────────────────────────────────────────────────────────

def load_conversation_logs(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        logs = json.load(f)
    logger.info("Loaded %d conversation logs from %s", len(logs), path)
    return logs


def load_questions_index(questions_path: Path) -> dict[str, dict]:
    """Returns {question_id: question_dict}"""
    with open(questions_path, "r", encoding="utf-8") as f:
        qs = json.load(f)
    return {q["id"]: q for q in qs}


# ── trial extraction ─────────────────────────────────────────────────────────

def extract_trial_context(log: dict) -> Optional[dict]:
    """
    From a conversation log, extract the information needed for MEBS analysis.

    Returns a dict with:
      experiment_id, model, question_id, question_text,
      ground_truth, false_claim, category, pressure_type,
      answer_t0, answer_t3, answer_t4 (if present),
      changed_t0_t3, changed_t3_t4
    """
    turns_by_idx = {t["turn"]: t for t in log.get("turns", [])}

    t0 = turns_by_idx.get(0)
    t3 = turns_by_idx.get(3)
    if t0 is None or t3 is None:
        return None

    answer_t0 = t0.get("answer", "")
    answer_t3 = t3.get("answer", "")

    t4 = turns_by_idx.get(4)
    answer_t4 = t4.get("answer", "") if t4 else None

    return {
        "experiment_id":  log["experiment_id"],
        "model":          log["model"],
        "question_id":    log["question_id"],
        "question_text":  None,          # filled later from questions index
        "ground_truth":   log["ground_truth"],
        "false_claim":    log["false_claim"],
        "category":       log["category"],
        "pressure_type":  log["pressure_type"],
        "answer_t0":      answer_t0,
        "answer_t3":      answer_t3,
        "answer_t4":      answer_t4,
        "changed_t0_t3":  answer_t0.strip().lower() != answer_t3.strip().lower(),
        "changed_t3_t4":  (
            answer_t4 is not None
            and answer_t3.strip().lower() != answer_t4.strip().lower()
        ),
        "turns_by_idx":   turns_by_idx,
    }


def determine_true_cause(
    ctx: dict,
    change_type: str,   # "t0_t3" or "t3_t4"
) -> CauseLabel:
    """
    Assign ground-truth cause label to an answer change.

    t0→t3 changes: the only thing that happened was social pressure → "social"
    t3→t4 changes: evidence was presented → "epistemic"
    """
    if change_type == "t0_t3":
        return "social"
    return "epistemic"


# ── MEBS runner ──────────────────────────────────────────────────────────────

class MEBSRunner:
    """
    Runs introspection probes on completed EPP experiment results.

    For every trial where the model changed its answer (T0→T3 or T3→T4),
    opens a fresh conversation and asks the model to explain why.
    """

    def __init__(
        self,
        results_dir:    Path,
        questions_path: Path,
        model_configs:  list[dict],
        seed:           int   = 42,
        max_retries:    int   = 3,
        backoff_base:   float = 2.0,
        timeout:        float = 30.0,
        probe_both_changes: bool = True,   # probe T3→T4 changes too
        max_probes_per_model: Optional[int] = None,   # cap for cost control
    ):
        self.results_dir  = results_dir
        self.questions    = load_questions_index(questions_path)
        self.model_cfgs   = {cfg["id"]: cfg for cfg in model_configs}
        self.seed         = seed
        self.max_retries  = max_retries
        self.backoff_base = backoff_base
        self.timeout      = timeout
        self.probe_both   = probe_both_changes
        self.max_probes   = max_probes_per_model

        self.output_dir   = results_dir / "mebs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.introspections: list[IntrospectionResult] = []
        self.trial_records:  list[MEBSTrialRecord]     = []

    def run(self) -> dict[str, MEBSMetrics]:
        logs_path = self.results_dir / "conversation_logs.json"
        if not logs_path.exists():
            raise FileNotFoundError(
                f"conversation_logs.json not found at {logs_path}. "
                "Run the EPP experiment first."
            )

        logs = load_conversation_logs(logs_path)
        contexts = [c for log in logs if (c := extract_trial_context(log)) is not None]

        # Attach question text from index
        for ctx in contexts:
            q = self.questions.get(ctx["question_id"], {})
            ctx["question_text"] = q.get("question", ctx["question_id"])

        # Build probe queue
        probes = self._build_probe_queue(contexts)
        logger.info("Probe queue: %d introspection calls to make.", len(probes))

        # Run probes
        with tqdm(total=len(probes), desc="Running introspection probes", unit="probe") as pbar:
            for probe in probes:
                result = run_introspection(
                    model_id      = probe["model_id"],
                    question_text = probe["question_text"],
                    answer_t0     = probe["answer_before"],
                    answer_t3     = probe["answer_after"],
                    experiment_id = probe["experiment_id"],
                    question_id   = probe["question_id"],
                    true_cause    = probe["true_cause"],
                    template_idx  = probe["template_idx"],
                    max_tokens    = self.model_cfgs.get(probe["model_id"], {}).get("max_tokens", 256),
                    temperature   = self.model_cfgs.get(probe["model_id"], {}).get("temperature", 0.0),
                    max_retries   = self.max_retries,
                    backoff_base  = self.backoff_base,
                    timeout       = self.timeout,
                )
                self.introspections.append(result)
                probe["introspection"] = result
                pbar.update(1)

        # Build MEBSTrialRecords
        self.trial_records = self._build_trial_records(contexts, probes)

        # Compute metrics per model
        model_metrics = {}
        models = sorted(set(r.model for r in self.trial_records))
        for model in models:
            model_recs = [r for r in self.trial_records if r.model == model]
            model_metrics[model] = compute_mebs_metrics(model_recs, model)

        self._save(model_metrics)
        self._print_summary(model_metrics)
        return model_metrics

    def _build_probe_queue(self, contexts: list[dict]) -> list[dict]:
        """
        Determine which (context, change_type) pairs need an introspection probe.
        """
        import random
        rng = random.Random(self.seed)

        probes_by_model: dict[str, list[dict]] = defaultdict(list)

        for ctx in contexts:
            # T0→T3 social pressure changes
            if ctx["changed_t0_t3"]:
                probes_by_model[ctx["model"]].append({
                    "experiment_id": ctx["experiment_id"],
                    "model_id":      ctx["model"],
                    "question_id":   ctx["question_id"],
                    "question_text": ctx["question_text"],
                    "answer_before": ctx["answer_t0"],
                    "answer_after":  ctx["answer_t3"],
                    "true_cause":    "social",
                    "change_type":   "t0_t3",
                    "template_idx":  rng.randint(0, len(probes_by_model) + 2),
                    "ctx":           ctx,
                })

            # T3→T4 evidence changes (optional)
            if self.probe_both and ctx.get("changed_t3_t4") and ctx.get("answer_t4"):
                probes_by_model[ctx["model"]].append({
                    "experiment_id": ctx["experiment_id"],
                    "model_id":      ctx["model"],
                    "question_id":   ctx["question_id"],
                    "question_text": ctx["question_text"],
                    "answer_before": ctx["answer_t3"],
                    "answer_after":  ctx["answer_t4"],
                    "true_cause":    "epistemic",
                    "change_type":   "t3_t4",
                    "template_idx":  rng.randint(0, len(probes_by_model) + 2),
                    "ctx":           ctx,
                })

        # Apply per-model cap
        all_probes = []
        for model, model_probes in probes_by_model.items():
            if self.max_probes is not None:
                # Stratify: keep equal social/epistemic if possible
                social    = [p for p in model_probes if p["true_cause"] == "social"]
                epistemic = [p for p in model_probes if p["true_cause"] == "epistemic"]
                cap_each  = self.max_probes // 2
                social    = rng.sample(social,    min(len(social),    cap_each))
                epistemic = rng.sample(epistemic, min(len(epistemic), cap_each))
                model_probes = social + epistemic

            all_probes.extend(model_probes)

        return all_probes

    def _build_trial_records(
        self,
        contexts: list[dict],
        probes:   list[dict],
    ) -> list[MEBSTrialRecord]:
        """
        Merge context data with introspection results into MEBSTrialRecords.
        One record per (experiment_id, change_type) — not per original trial.
        """
        # Index probes by (experiment_id, change_type)
        probe_index: dict[tuple[str, str], dict] = {
            (p["experiment_id"], p["change_type"]): p
            for p in probes
        }

        records = []
        for ctx in contexts:
            gt  = ctx["ground_truth"]
            fc  = ctx["false_claim"]

            for change_type, answer_before, answer_after, changed in [
                ("t0_t3", ctx["answer_t0"], ctx["answer_t3"], ctx["changed_t0_t3"]),
                ("t3_t4", ctx["answer_t3"], ctx.get("answer_t4", ""),
                 ctx.get("changed_t3_t4", False)),
            ]:
                if change_type == "t3_t4" and not self.probe_both:
                    continue
                if change_type == "t3_t4" and not ctx.get("answer_t4"):
                    continue

                probe     = probe_index.get((ctx["experiment_id"], change_type))
                intro_res = probe["introspection"] if probe and "introspection" in probe else None

                sc  = float(answer_matches_false_claim(answer_after, fc))
                ec  = float(evaluate_answer(answer_after, gt))

                rec = MEBSTrialRecord(
                    experiment_id          = ctx["experiment_id"],
                    model                  = ctx["model"],
                    question_id            = ctx["question_id"],
                    category               = ctx["category"],
                    pressure_type          = ctx["pressure_type"],
                    answer_changed         = changed,
                    answer_t0              = answer_before,
                    answer_t3              = answer_after,
                    true_cause             = determine_true_cause(ctx, change_type) if changed else None,
                    introspection          = intro_res,
                    social_coherence_t3    = sc,
                    epistemic_coherence_t3 = ec,
                )
                records.append(rec)

        return records

    def _save(self, model_metrics: dict[str, MEBSMetrics]) -> None:
        """Persist all MEBS outputs."""

        # introspections CSV
        intro_path = self.output_dir / "mebs_introspections.csv"
        with open(intro_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=INTRO_FIELDNAMES)
            writer.writeheader()
            for ir in self.introspections:
                row = asdict(ir)
                writer.writerow({k: row[k] for k in INTRO_FIELDNAMES})
        logger.info("Saved %d introspections → %s", len(self.introspections), intro_path)

        # trial records CSV
        trial_path = self.output_dir / "mebs_trial_records.csv"
        with open(trial_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDNAMES)
            writer.writeheader()
            for r in self.trial_records:
                intro = r.introspection
                writer.writerow({
                    "experiment_id":          r.experiment_id,
                    "model":                  r.model,
                    "question_id":            r.question_id,
                    "category":               r.category,
                    "pressure_type":          r.pressure_type,
                    "answer_changed":         r.answer_changed,
                    "answer_t0":              r.answer_t0,
                    "answer_t3":              r.answer_t3,
                    "true_cause":             r.true_cause or "",
                    "attribution":            intro.attribution if intro else "",
                    "attribution_conf":       intro.attribution_conf if intro else "",
                    "explanation":            intro.explanation if intro else "",
                    "social_coherence_t3":    round(r.social_coherence_t3,    4),
                    "epistemic_coherence_t3": round(r.epistemic_coherence_t3, 4),
                })
        logger.info("Saved %d trial records → %s", len(self.trial_records), trial_path)

        # model metrics CSV
        metrics_path = self.output_dir / "mebs_model_metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
            writer.writeheader()
            for m, mm in sorted(model_metrics.items()):
                row = asdict(mm)
                writer.writerow({k: round(row[k], 4) if isinstance(row[k], float) else row[k]
                                  for k in METRICS_FIELDNAMES})
        logger.info("Saved model metrics → %s", metrics_path)

        # logs JSON
        logs_path = self.output_dir / "mebs_logs.json"
        logs_out  = []
        for ir in self.introspections:
            d = asdict(ir)
            logs_out.append(d)
        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(logs_out, f, indent=2, ensure_ascii=False)
        logger.info("Saved MEBS logs → %s", logs_path)

    @staticmethod
    def _print_summary(model_metrics: dict[str, MEBSMetrics]) -> None:
        import math
        def f(v):
            return "  N/A  " if (v is None or (isinstance(v, float) and math.isnan(v))) \
                   else f"{float(v):>6.3f}"

        header = (
            f"{'Model':<45} "
            f"{'MEBS':>6} {'Attr%':>6} {'SC':>6} "
            f"{'EC':>6} {'Gap':>6} {'Spec':>6}"
        )
        sep = "─" * len(header)
        print(f"\n{sep}\nMETA-EPISTEMIC BLINDSPOT SUMMARY\n{sep}")
        print(header)
        print(sep)
        for m, mm in sorted(model_metrics.items()):
            print(
                f"{m:<45} "
                f"{f(mm.mebs_score)} "
                f"{f(mm.attribution_accuracy)} "
                f"{f(mm.social_coherence)} "
                f"{f(mm.epistemic_coherence)} "
                f"{f(mm.sc_ec_gap)} "
                f"{f(mm.blindspot_specificity)}"
            )
        print(sep)
        print("\nMEBS=blindspot rate  Attr%=attribution accuracy")
        print("SC=social coherence  EC=epistemic coherence")
        print("Gap=SC-EC  Spec=blindspot specificity (higher=more self-aware)\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="MEBS Runner — Meta-Epistemic Blindspot experiment"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing results.csv and conversation_logs.json",
    )
    parser.add_argument(
        "--questions",
        default="data/questions.json",
        help="Path to questions.json",
    )
    parser.add_argument(
        "--models-config",
        default="config/models.json",
        help="Path to models.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-probes-per-model",
        type=int,
        default=None,
        help="Cap introspection probes per model (for cost control).",
    )
    parser.add_argument(
        "--no-evidence-probes",
        action="store_true",
        help="Only probe social-cause changes (T0→T3), skip T3→T4.",
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    with open(base / args.models_config, "r") as f:
        cfg = json.load(f)

    runner = MEBSRunner(
        results_dir          = base / args.results_dir,
        questions_path       = base / args.questions,
        model_configs        = cfg["models"],
        seed                 = args.seed,
        max_probes_per_model = args.max_probes_per_model,
        probe_both_changes   = not args.no_evidence_probes,
    )
    runner.run()


if __name__ == "__main__":
    main()
