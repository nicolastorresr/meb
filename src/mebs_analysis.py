"""
mebs_analysis.py
================
Statistical analysis for the Meta-Epistemic Blindspot experiment.

Inputs:
    results/mebs/mebs_introspections.csv
    results/mebs/mebs_trial_records.csv
    results/mebs/mebs_model_metrics.csv

Outputs:
    results/mebs/analysis/mebs_statistical_tests.json
    results/mebs/analysis/mebs_attribution_by_pressure.csv
    results/mebs/analysis/mebs_coherence_by_category.csv
    results/mebs/analysis/mebs_bootstrap_mebs_score.csv

Statistical tests
-----------------
1. Binomial test per model
   H0: P(epistemic attribution | social cause) = 0.5
   — tests whether the blindspot rate is significantly above chance.

2. McNemar test per model (paired)
   Compares P(epistemic attr | social) vs P(epistemic attr | epistemic)
   — tests whether the model distinguishes causes at all.

3. Chi-square: attribution distribution differs across models?

4. Chi-square: attribution distribution differs across pressure types?

5. Bootstrap CIs for MEBS score (n=1000).

6. Logistic regression: predict P(epistemic attribution)
   from [true_cause, model, pressure_type, question_category]
   — identifies what drives the blindspot.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_trial_records(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["answer_changed"]         = row["answer_changed"].lower() == "true"
        row["social_coherence_t3"]    = float(row["social_coherence_t3"])
        row["epistemic_coherence_t3"] = float(row["epistemic_coherence_t3"])
        try:
            row["attribution_conf"] = int(row["attribution_conf"]) if row["attribution_conf"] else None
        except ValueError:
            row["attribution_conf"] = None
    logger.info("Loaded %d MEBS trial records from %s", len(rows), path)
    return rows


def load_introspections(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["parse_ok"] = row["parse_ok"].lower() == "true"
        try:
            row["attribution_conf"] = int(row["attribution_conf"]) if row["attribution_conf"] else None
        except ValueError:
            row["attribution_conf"] = None
    logger.info("Loaded %d introspection records from %s", len(rows), path)
    return rows


def load_model_metrics(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    float_fields = [
        "mebs_score", "attribution_accuracy",
        "social_coherence", "epistemic_coherence", "sc_ec_gap",
        "blindspot_specificity",
        "p_epistemic_given_social", "p_epistemic_given_epistemic",
        "p_social_given_social", "p_social_given_epistemic",
    ]
    int_fields = ["n_trials", "n_changed"]
    for row in rows:
        for f in float_fields:
            try:
                row[f] = float(row[f]) if row[f] not in ("", "nan") else float("nan")
            except ValueError:
                row[f] = float("nan")
        for f in int_fields:
            try:
                row[f] = int(row[f])
            except ValueError:
                row[f] = 0
    return rows


# ── helper: valid introspection rows ─────────────────────────────────────────

def _valid(rows: list[dict]) -> list[dict]:
    """Rows with a non-empty, non-parse_error attribution and a true_cause."""
    return [
        r for r in rows
        if r.get("true_cause") in ("social", "epistemic")
        and r.get("attribution") not in ("", "parse_error", None)
    ]


def _p(rows: list[dict], attr: str) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r["attribution"] == attr) / len(rows)


# ── 1. Binomial test per model ────────────────────────────────────────────────

def binomial_test_mebs(trial_rows: list[dict]) -> dict[str, dict]:
    """
    Per model: test H0: P(ep attr | social cause) = 0.5 (two-sided).
    A significant result with p > 0.5 means the blindspot is real.
    """
    models  = sorted(set(r["model"] for r in trial_rows))
    results = {}
    for model in models:
        social_rows = [
            r for r in trial_rows
            if r["model"] == model
            and r["true_cause"] == "social"
            and r["attribution"] not in ("", "parse_error", None)
        ]
        n = len(social_rows)
        k = sum(1 for r in social_rows if r["attribution"] == "epistemic")
        if n == 0:
            results[model] = {"n": 0, "k_epistemic": 0,
                               "p_epistemic": float("nan"), "p_value": float("nan")}
            continue
        p_obs = k / n
        # scipy binomtest
        bt  = stats.binomtest(k, n, p=0.5, alternative="two-sided")
        results[model] = {
            "n_social_cause_trials": n,
            "k_epistemic_attribution": k,
            "p_epistemic_observed": round(p_obs, 4),
            "binomial_p_value": round(float(bt.pvalue), 4),
            "significant_blindspot": bt.pvalue < 0.05 and p_obs > 0.5,
        }
    return results


# ── 2. McNemar test per model ─────────────────────────────────────────────────

def mcnemar_test_mebs(trial_rows: list[dict]) -> dict[str, dict]:
    """
    Per model: does the model attribute changes differently for
    social-cause vs epistemic-cause trials?

    Paired design: count
      a = social-cause, said "epistemic"
      b = social-cause, said "social"
      c = epistemic-cause, said "epistemic"
      d = epistemic-cause, said "social"

    McNemar on the 2×2 table of attribution for the two cause types.
    Since samples are unpaired (different trials), we use chi-square
    comparing proportions instead — reported as a Fisher exact test.
    """
    models  = sorted(set(r["model"] for r in trial_rows))
    results = {}
    for model in models:
        valid = _valid([r for r in trial_rows if r["model"] == model])
        social_rows    = [r for r in valid if r["true_cause"] == "social"]
        epistemic_rows = [r for r in valid if r["true_cause"] == "epistemic"]

        if not social_rows or not epistemic_rows:
            results[model] = {"error": "Insufficient data for both cause types."}
            continue

        # 2×2 contingency: rows=cause, cols=attribution(ep vs not)
        a = sum(1 for r in social_rows    if r["attribution"] == "epistemic")
        b = len(social_rows) - a
        c = sum(1 for r in epistemic_rows if r["attribution"] == "epistemic")
        d = len(epistemic_rows) - c

        table = [[a, b], [c, d]]
        odds_ratio, p_value = stats.fisher_exact(table, alternative="two-sided")
        results[model] = {
            "social_cause_n":        len(social_rows),
            "epistemic_cause_n":     len(epistemic_rows),
            "contingency_table":     table,
            "p_epistemic_attr_given_social":    round(a / len(social_rows), 4),
            "p_epistemic_attr_given_epistemic": round(c / len(epistemic_rows), 4),
            "fisher_odds_ratio":     round(float(odds_ratio), 4),
            "fisher_p_value":        round(float(p_value), 4),
            "distinguishes_causes":  p_value < 0.05,
        }
    return results


# ── 3. Chi-square: attributions differ across models ─────────────────────────

def chi_square_attribution_across_models(trial_rows: list[dict]) -> dict:
    """
    Chi-square test: does the attribution distribution (ep/social/mixed/uncertain)
    differ significantly across models for social-cause trials?
    """
    valid = _valid([r for r in trial_rows if r["true_cause"] == "social"])
    if not valid:
        return {"error": "No valid social-cause attribution rows."}

    models = sorted(set(r["model"] for r in valid))
    labels = ["epistemic", "social", "mixed", "uncertain"]

    table = []
    for model in models:
        model_rows = [r for r in valid if r["model"] == model]
        table.append([
            sum(1 for r in model_rows if r["attribution"] == lbl)
            for lbl in labels
        ])

    arr = np.array(table)
    # Drop columns that are all-zero
    nonzero_cols = [j for j in range(arr.shape[1]) if arr[:, j].sum() > 0]
    arr = arr[:, nonzero_cols]
    active_labels = [labels[j] for j in nonzero_cols]

    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return {"error": "Insufficient variation for chi-square test."}

    chi2, p, dof, _ = stats.chi2_contingency(arr)
    return {
        "test":          "chi2_contingency",
        "chi2":          round(float(chi2), 4),
        "p_value":       round(float(p), 4),
        "dof":           int(dof),
        "models":        models,
        "labels":        active_labels,
        "table":         arr.tolist(),
        "significant":   p < 0.05,
    }


# ── 4. Chi-square: attributions differ across pressure types ─────────────────

def chi_square_attribution_by_pressure(trial_rows: list[dict]) -> dict:
    """
    Does the attribution distribution differ across pressure types?
    Tests whether some pressure types are better at inducing blindspots.
    """
    valid = _valid([r for r in trial_rows if r["true_cause"] == "social"])
    if not valid:
        return {"error": "No valid social-cause rows."}

    pts    = sorted(set(r["pressure_type"] for r in valid))
    labels = ["epistemic", "social", "mixed", "uncertain"]

    table = []
    for pt in pts:
        pt_rows = [r for r in valid if r["pressure_type"] == pt]
        table.append([sum(1 for r in pt_rows if r["attribution"] == lbl) for lbl in labels])

    arr = np.array(table)
    nonzero_cols = [j for j in range(arr.shape[1]) if arr[:, j].sum() > 0]
    arr = arr[:, nonzero_cols]
    active_labels = [labels[j] for j in nonzero_cols]

    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return {"error": "Insufficient variation."}

    chi2, p, dof, _ = stats.chi2_contingency(arr)
    return {
        "test":           "chi2_contingency",
        "chi2":           round(float(chi2), 4),
        "p_value":        round(float(p), 4),
        "dof":            int(dof),
        "pressure_types": pts,
        "labels":         active_labels,
        "table":          arr.tolist(),
        "significant":    p < 0.05,
    }


# ── 5. Bootstrap CIs for MEBS score ──────────────────────────────────────────

def bootstrap_mebs_score(
    trial_rows: list[dict],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict]:
    """Bootstrap confidence intervals for MEBS score per model."""
    rng    = np.random.default_rng(seed)
    models = sorted(set(r["model"] for r in trial_rows))
    result = {}

    for model in models:
        social_rows = [
            r for r in trial_rows
            if r["model"] == model
            and r["true_cause"] == "social"
            and r["attribution"] not in ("", "parse_error", None)
        ]
        n = len(social_rows)
        if n == 0:
            result[model] = {"mebs_score": float("nan"),
                             "ci_lower": float("nan"), "ci_upper": float("nan")}
            continue

        boot_scores = []
        for _ in range(n_boot):
            sample = [social_rows[i] for i in rng.integers(0, n, size=n)]
            k = sum(1 for r in sample if r["attribution"] == "epistemic")
            boot_scores.append(k / n)

        alpha = 1.0 - ci
        point = sum(1 for r in social_rows if r["attribution"] == "epistemic") / n
        result[model] = {
            "mebs_score": round(point, 4),
            "ci_lower":   round(float(np.percentile(boot_scores, 100 * alpha / 2)), 4),
            "ci_upper":   round(float(np.percentile(boot_scores, 100 * (1 - alpha / 2))), 4),
        }
    return result


# ── 6. Logistic regression: what predicts epistemic attribution? ──────────────

def logistic_regression_blindspot(trial_rows: list[dict]) -> dict:
    """
    Logistic regression predicting P(attribution = 'epistemic')
    from [true_cause, model, pressure_type, category].

    Uses scipy.stats.logistic — or falls back to a simple odds-ratio
    table if sklearn is not available.
    """
    valid = _valid(trial_rows)
    if len(valid) < 20:
        return {"error": "Insufficient data for logistic regression (need ≥ 20 rows)."}

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd

        df = pd.DataFrame(valid)
        df["y"] = (df["attribution"] == "epistemic").astype(int)

        feature_cols = ["true_cause", "model", "pressure_type", "category"]
        for col in feature_cols:
            if col not in df.columns:
                df[col] = "unknown"

        # One-hot encode
        X = pd.get_dummies(df[feature_cols], drop_first=True).astype(float)
        y = df["y"].values

        if len(np.unique(y)) < 2:
            return {"error": "No variation in outcome (attribution)."}

        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X, y)

        coef_dict = dict(zip(X.columns.tolist(), lr.coef_[0].tolist()))
        return {
            "method":     "logistic_regression_sklearn",
            "n_samples":  int(len(y)),
            "intercept":  round(float(lr.intercept_[0]), 4),
            "coefficients": {k: round(v, 4) for k, v in coef_dict.items()},
            "feature_names": X.columns.tolist(),
            "note": (
                "Positive coefficient = factor increases P(epistemic attribution). "
                "true_cause_social: positive = social cause → more epistemic attribution = blindspot."
            ),
        }
    except ImportError:
        # Fallback: simple odds ratio table
        social_ep    = sum(1 for r in valid if r["true_cause"] == "social"    and r["attribution"] == "epistemic")
        social_not   = sum(1 for r in valid if r["true_cause"] == "social"    and r["attribution"] != "epistemic")
        ep_ep        = sum(1 for r in valid if r["true_cause"] == "epistemic" and r["attribution"] == "epistemic")
        ep_not       = sum(1 for r in valid if r["true_cause"] == "epistemic" and r["attribution"] != "epistemic")
        # Add 0.5 Haldane correction
        or_val = ((social_ep + 0.5) * (ep_not + 0.5)) / ((social_not + 0.5) * (ep_ep + 0.5))
        return {
            "method": "odds_ratio_fallback",
            "note":   "sklearn not available; reporting simple odds ratio.",
            "odds_ratio_social_vs_epistemic_cause": round(float(or_val), 4),
            "interpretation": (
                "OR > 1 means social-cause trials more likely to receive 'epistemic' attribution "
                "(blindspot present). OR = 1 means no blindspot."
            ),
        }


# ── Sliced attribution tables ─────────────────────────────────────────────────

def attribution_by_pressure_type(trial_rows: list[dict]) -> list[dict]:
    """
    For each (model, pressure_type), compute:
      p_epistemic_attr, p_social_attr, n_social_cause_trials
    Only for social-cause trials.
    """
    valid = _valid([r for r in trial_rows if r["true_cause"] == "social"])
    models = sorted(set(r["model"] for r in valid))
    pts    = sorted(set(r["pressure_type"] for r in valid))
    rows   = []
    for model in models:
        for pt in pts:
            subset = [r for r in valid if r["model"] == model and r["pressure_type"] == pt]
            if not subset:
                continue
            rows.append({
                "model":          model,
                "pressure_type":  pt,
                "n":              len(subset),
                "p_epistemic":    round(_p(subset, "epistemic"), 4),
                "p_social":       round(_p(subset, "social"),    4),
                "p_mixed":        round(_p(subset, "mixed"),     4),
                "p_uncertain":    round(_p(subset, "uncertain"), 4),
            })
    return rows


def coherence_by_category(trial_rows: list[dict]) -> list[dict]:
    """
    For each (model, category), compute SC, EC, and SC-EC gap.
    """
    models = sorted(set(r["model"] for r in trial_rows))
    cats   = sorted(set(r["category"] for r in trial_rows))
    rows   = []
    for model in models:
        for cat in cats:
            subset = [r for r in trial_rows
                      if r["model"] == model and r["category"] == cat]
            if not subset:
                continue
            sc = sum(r["social_coherence_t3"]    for r in subset) / len(subset)
            ec = sum(r["epistemic_coherence_t3"] for r in subset) / len(subset)
            rows.append({
                "model":               model,
                "category":            cat,
                "n":                   len(subset),
                "social_coherence":    round(sc, 4),
                "epistemic_coherence": round(ec, 4),
                "sc_ec_gap":           round(sc - ec, 4),
            })
    return rows


# ── Export helpers ────────────────────────────────────────────────────────────

def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_mebs_analysis(mebs_dir: str | Path, output_dir: Optional[str | Path] = None) -> dict:
    """
    Full MEBS statistical analysis pipeline.

    Args:
        mebs_dir:   Path to results/mebs/ (contains the three CSV files).
        output_dir: Where to save analysis outputs. Defaults to mebs_dir/analysis/.

    Returns:
        Dict with all computed analysis results.
    """
    mebs_dir = Path(mebs_dir)
    out      = Path(output_dir) if output_dir else mebs_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    trial_rows   = load_trial_records(mebs_dir / "mebs_trial_records.csv")
    introspections = load_introspections(mebs_dir / "mebs_introspections.csv")

    # Statistical tests
    binom     = binomial_test_mebs(trial_rows)
    mcnemar   = mcnemar_test_mebs(trial_rows)
    chi2_m    = chi_square_attribution_across_models(trial_rows)
    chi2_pt   = chi_square_attribution_by_pressure(trial_rows)
    boot      = bootstrap_mebs_score(trial_rows)
    logreg    = logistic_regression_blindspot(trial_rows)

    stats_out = {
        "binomial_test_per_model":                binom,
        "fisher_exact_cause_discrimination":      mcnemar,
        "chi2_attribution_across_models":         chi2_m,
        "chi2_attribution_by_pressure_type":      chi2_pt,
        "bootstrap_mebs_score":                   boot,
        "logistic_regression_blindspot_predictor": logreg,
    }
    class _SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):           return int(obj)
            if isinstance(obj, (np.floating,)):          return float(obj)
            if isinstance(obj, (np.bool_,)):             return bool(obj)
            if isinstance(obj, (np.ndarray,)):           return obj.tolist()
            if isinstance(obj, float) and math.isnan(obj): return None
            return super().default(obj)
        def encode(self, obj):
            # walk the tree and convert Python bools / NaN floats safely
            return super().encode(self._sanitize(obj))
        def _sanitize(self, obj):
            if isinstance(obj, dict):
                return {k: self._sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [self._sanitize(v) for v in obj]
            if isinstance(obj, float) and math.isnan(obj):
                return None
            return obj

    with open(out / "mebs_statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, cls=_SafeEncoder)
    logger.info("Saved statistical tests → %s", out / "mebs_statistical_tests.json")

    # Bootstrap CSV
    boot_rows = [{"model": m, **v} for m, v in sorted(boot.items())]
    _save_csv(boot_rows, out / "mebs_bootstrap_mebs_score.csv")

    # Sliced tables
    attr_by_pt = attribution_by_pressure_type(trial_rows)
    _save_csv(attr_by_pt, out / "mebs_attribution_by_pressure.csv")

    coh_by_cat = coherence_by_category(trial_rows)
    _save_csv(coh_by_cat, out / "mebs_coherence_by_category.csv")

    logger.info("MEBS analysis complete → %s", out)

    return {
        "trial_rows":        trial_rows,
        "introspections":    introspections,
        "statistical_tests": stats_out,
        "attribution_by_pressure": attr_by_pt,
        "coherence_by_category":   coh_by_cat,
        "bootstrap":         boot,
    }
