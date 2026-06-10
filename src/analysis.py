"""
analysis.py
Full analysis pipeline for the EPP experiment.

Outputs:
  - Per-model, per-pressure-type, per-category metrics (CSV)
  - Compliance chi-square test
  - Confidence drift ANOVA
  - Bootstrap CIs for Stability
  - Version Stability analysis (inter-family vs intra-family EPP distance)
  - Statistical test results (JSON)
"""

import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from metrics import (
    EPP_DIMENSIONS,
    EpistemicMetrics,
    TrialRecord,
    compute_metrics,
    normalize_profile,
    EVIDENCE_TURN,
    MAX_PRESSURE_TURN,
)

logger = logging.getLogger(__name__)


# ─── CSV loading ─────────────────────────────────────────────────────────────

def load_results_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["turn"]       = int(row["turn"])
        row["confidence"] = int(row["confidence"]) if row["confidence"] else 50
        row["correct"]    = row["correct"].lower() == "true"
        row["adopted_false_claim"] = row["adopted_false_claim"].lower() == "true"
        row["reasoning"]  = row.get("reasoning", "")
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


# ─── trial grouping ──────────────────────────────────────────────────────────

def rows_to_trial_records(rows: list[dict]) -> list[TrialRecord]:
    """Convert flat CSV rows to TrialRecord objects."""
    by_experiment: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment_id"]].append(row)

    records = []
    for eid, turns in by_experiment.items():
        turns.sort(key=lambda r: r["turn"])
        t0 = turns[0]
        rec = TrialRecord(
            model=t0["model"],
            question_id=t0["question_id"],
            category=t0["category"],
            pressure_type=t0["pressure_type"],
        )
        for t in turns:
            turn = t["turn"]
            rec.answers[turn]             = t["answer"]
            rec.confidences[turn]         = t["confidence"]
            rec.reasonings[turn]          = t.get("reasoning", "")
            rec.correct[turn]             = t["correct"]
            rec.adopted_false_claim[turn] = t["adopted_false_claim"]
        records.append(rec)
    return records


# ─── sliced metric computation ───────────────────────────────────────────────

def compute_model_metrics(records: list[TrialRecord]) -> dict[str, dict]:
    models = sorted(set(r.model for r in records))
    return {
        m: _metrics_to_dict(compute_metrics([r for r in records if r.model == m]))
        for m in models
    }


def compute_pressure_metrics(records: list[TrialRecord]) -> dict[str, dict]:
    pts = sorted(set(r.pressure_type for r in records))
    return {
        pt: _metrics_to_dict(compute_metrics([r for r in records if r.pressure_type == pt]))
        for pt in pts
    }


def compute_category_metrics(records: list[TrialRecord]) -> dict[str, dict]:
    cats = sorted(set(r.category for r in records))
    return {
        cat: _metrics_to_dict(compute_metrics([r for r in records if r.category == cat]))
        for cat in cats
    }


def compute_heatmap_data(records: list[TrialRecord]) -> dict[str, dict[str, float]]:
    """Compliance heatmap: {model: {pressure_type: compliance}}"""
    models = sorted(set(r.model for r in records))
    pts    = sorted(set(r.pressure_type for r in records))
    result = {}
    for m in models:
        result[m] = {}
        for pt in pts:
            subset = [r for r in records if r.model == m and r.pressure_type == pt]
            result[m][pt] = _metrics_to_dict(compute_metrics(subset))["compliance"]
    return result


def _metrics_to_dict(m: EpistemicMetrics) -> dict:
    return {
        "n_trials":             m.n_trials,
        "stability":            m.stability,
        "compliance":           m.compliance,
        "confidence_drift_mean":m.confidence_drift_mean,
        "confidence_drift_std": m.confidence_drift_std,
        "flexibility":          m.flexibility,
        "recovery":             m.recovery,
        "self_awareness":       m.self_awareness,
    }


# ─── statistical tests ───────────────────────────────────────────────────────

def chi_square_compliance(records: list[TrialRecord]) -> dict:
    """Chi-square test: compliance rates differ across models?"""
    models = sorted(set(r.model for r in records))
    observed = []
    for m in models:
        subset = [r for r in records if r.model == m]
        n = len(subset)
        compliant = sum(
            1 for r in subset
            if any(r.adopted_false_claim.get(t, False) for t in range(1, MAX_PRESSURE_TURN + 1))
        )
        observed.append([compliant, n - compliant])

    table = np.array(observed)
    if table.shape[0] < 2:
        return {"error": "Not enough groups."}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"test": "chi2_contingency", "chi2": float(chi2), "p_value": float(p),
            "dof": int(dof), "models": models, "observed": table.tolist()}


def anova_confidence_drift(records: list[TrialRecord]) -> dict:
    """One-way ANOVA: confidence drift differs across models?"""
    models = sorted(set(r.model for r in records))
    groups = []
    for m in models:
        drifts = []
        for r in records:
            if r.model != m:
                continue
            c0 = r.confidences.get(0)
            c3 = r.confidences.get(MAX_PRESSURE_TURN)
            if c0 is not None and c3 is not None:
                drifts.append(abs(c3 - c0))
        groups.append(drifts)

    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return {"error": "Not enough groups."}
    f, p = stats.f_oneway(*groups)
    return {"test": "one_way_anova", "f_statistic": float(f), "p_value": float(p),
            "models": models, "group_sizes": [len(g) for g in groups]}


def bootstrap_stability(
    records: list[TrialRecord],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict]:
    """Bootstrap CIs for Stability per model."""
    from metrics import compute_stability
    rng    = np.random.default_rng(seed)
    models = sorted(set(r.model for r in records))
    result = {}
    for m in models:
        subset = [r for r in records if r.model == m]
        n = len(subset)
        boots = []
        for _ in range(n_boot):
            sample = [subset[i] for i in rng.integers(0, n, size=n)]
            v = compute_stability(sample)
            if not math.isnan(v):
                boots.append(v)
        if not boots:
            result[m] = {"stability": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
            continue
        alpha = 1.0 - ci
        point = compute_stability(subset)
        result[m] = {
            "stability": point,
            "ci_lower": float(np.percentile(boots, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        }
    return result


# ─── Version Stability analysis ──────────────────────────────────────────────

def _epp_vector(profile: dict[str, float]) -> np.ndarray:
    """Return a numeric vector for the 6 EPP dimensions (NaN → 0.5)."""
    return np.array(
        [0.5 if math.isnan(profile.get(d, float("nan"))) else profile.get(d, 0.5)
         for d in EPP_DIMENSIONS],
        dtype=float,
    )


def _euclidean(a: dict, b: dict) -> float:
    return float(np.linalg.norm(_epp_vector(a) - _epp_vector(b)))


def version_stability_analysis(
    normalized_profiles: dict[str, dict[str, float]],
    model_families: Optional[dict[str, str]] = None,
) -> dict:
    """
    Test hypothesis: D(same_family) < D(different_family).

    Args:
        normalized_profiles: {model_id: {dim: value}} — already normalized to [0,1].
        model_families:       {model_id: family_name}
                              If None, families are inferred from the provider prefix
                              (everything before '/').

    Returns:
        dict with intra-family distances, inter-family distances, and a
        Mann-Whitney U test result.
    """
    models = list(normalized_profiles.keys())
    if len(models) < 2:
        return {"error": "Need at least 2 models."}

    if model_families is None:
        model_families = {m: m.split("/")[0] for m in models}

    intra_distances = []
    inter_distances = []
    pair_details    = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            d = _euclidean(normalized_profiles[ma], normalized_profiles[mb])
            same_family = model_families.get(ma) == model_families.get(mb)
            pair_details.append({
                "model_a": ma, "model_b": mb,
                "family_a": model_families.get(ma, "?"),
                "family_b": model_families.get(mb, "?"),
                "same_family": same_family,
                "euclidean_distance": round(d, 4),
            })
            if same_family:
                intra_distances.append(d)
            else:
                inter_distances.append(d)

    result: dict = {
        "pair_details": pair_details,
        "intra_family_mean": float(np.mean(intra_distances)) if intra_distances else float("nan"),
        "inter_family_mean": float(np.mean(inter_distances)) if inter_distances else float("nan"),
        "hypothesis": "D(same_family) < D(different_family)",
    }

    # Mann-Whitney U test (one-sided: intra < inter)
    if len(intra_distances) >= 1 and len(inter_distances) >= 1:
        stat, p = stats.mannwhitneyu(
            intra_distances, inter_distances, alternative="less"
        )
        result["mann_whitney_u"] = {
            "statistic": float(stat),
            "p_value": float(p),
            "alternative": "intra < inter",
            "hypothesis_supported": p < 0.05,
        }
    else:
        result["mann_whitney_u"] = {
            "note": "Insufficient data for Mann-Whitney U (need same-family pairs)."
        }

    return result


# ─── export helpers ──────────────────────────────────────────────────────────

def export_dict_to_csv(data: dict, path: Path, key_label: str = "key") -> None:
    if not data:
        return
    skip = {"drifts"}
    cols = [key_label] + sorted({k for v in data.values() for k in v if k not in skip})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for key, metrics in sorted(data.items()):
            row = {key_label: key}
            for k, v in metrics.items():
                if k in skip:
                    continue
                row[k] = round(v, 4) if isinstance(v, float) else v
            writer.writerow(row)


# ─── main pipeline ───────────────────────────────────────────────────────────

def run_analysis(
    results_csv: str,
    output_dir: str,
    model_families: Optional[dict[str, str]] = None,
) -> dict:
    """
    Full analysis pipeline.

    Args:
        results_csv:    Path to results.csv.
        output_dir:     Directory for analysis outputs.
        model_families: Optional {model_id: family} for Version Stability.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows    = load_results_csv(results_csv)
    records = rows_to_trial_records(rows)

    model_metrics    = compute_model_metrics(records)
    pressure_metrics = compute_pressure_metrics(records)
    category_metrics = compute_category_metrics(records)
    heatmap_data     = compute_heatmap_data(records)

    chi2_result   = chi_square_compliance(records)
    anova_result  = anova_confidence_drift(records)
    boot_result   = bootstrap_stability(records)

    # Version Stability
    raw_profiles = {
        m: {d: model_metrics[m].get(d, float("nan")) for d in EPP_DIMENSIONS}
        for m in model_metrics
    }
    normalized = normalize_profile(raw_profiles)
    vs_result  = version_stability_analysis(normalized, model_families)

    # Exports
    export_dict_to_csv(model_metrics,    out / "model_metrics.csv",    "model")
    export_dict_to_csv(pressure_metrics, out / "pressure_metrics.csv", "pressure_type")
    export_dict_to_csv(category_metrics, out / "category_metrics.csv", "category")

    stats_out = {
        "chi2_compliance":        chi2_result,
        "anova_confidence_drift": anova_result,
        "bootstrap_stability":    boot_result,
        "version_stability":      vs_result,
    }
    with open(out / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2)

    with open(out / "bootstrap_stability.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "stability", "ci_lower", "ci_upper"])
        w.writeheader()
        for m, v in sorted(boot_result.items()):
            w.writerow({"model": m, **v})

    # Version Stability pair distances CSV
    if "pair_details" in vs_result:
        with open(out / "version_stability_pairs.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = ["model_a", "model_b", "family_a", "family_b",
                          "same_family", "euclidean_distance"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(vs_result["pair_details"])

    logger.info("Analysis complete → %s", out)

    return {
        "rows":             rows,
        "records":          records,
        "model_metrics":    model_metrics,
        "pressure_metrics": pressure_metrics,
        "category_metrics": category_metrics,
        "heatmap_data":     heatmap_data,
        "normalized_profiles": normalized,
        "statistical_tests": stats_out,
    }
