"""
mebs_visualization.py
=====================
Publication-quality figures for the Meta-Epistemic Blindspot experiment.

Figure M1 — MEBS Score by model (bar + bootstrap CI)
             "Blindspot rate: P(epistemic attribution | social cause)"

Figure M2 — Attribution confusion matrix per model
             Rows: true_cause (social / epistemic)
             Cols: model attribution (epistemic / social / mixed / uncertain)

Figure M3 — Social Coherence vs Epistemic Coherence scatter
             One point per model; SC-EC gap annotated

Figure M4 — Attribution by pressure type heatmap
             Rows: models, Cols: pressure types, Values: P(epistemic attr)

Figure M5 — Blindspot Specificity vs MEBS Score scatter
             The key diagnostic plot: low specificity + high MEBS = deepest blindspot
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

DPI        = 300
SIZE_BAR   = (8, 5)
SIZE_CONF  = (10, 4)
SIZE_SCAT  = (7, 6)
SIZE_HEAT  = (10, 5)
SIZE_DIAG  = (7, 6)

PALETTE = ["#2196F3","#FF5722","#4CAF50","#9C27B0",
           "#FF9800","#00BCD4","#E91E63","#607D8B"]

FONT_T  = {"fontsize": 14, "fontweight": "bold"}
FONT_L  = {"fontsize": 11}
FONT_TK = {"fontsize": 9}

ATTR_COLORS = {
    "epistemic": "#E53935",   # red — wrong attribution
    "social":    "#43A047",   # green — correct attribution
    "mixed":     "#FB8C00",   # orange
    "uncertain": "#8E9EAB",   # grey
}
CAUSE_LABELS = {"social": "Social cause\n(pressure turns)", "epistemic": "Epistemic cause\n(evidence turn)"}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


def _short(model_id: str) -> str:
    MAP = {
        "openai/gpt-4o":                     "GPT-4o",
        "openai/gpt-4.1":                    "GPT-4.1",
        "anthropic/claude-sonnet-4-5":       "Claude S4.5",
        "anthropic/claude-sonnet-4.5":       "Claude S4.5",
        "anthropic/claude-opus-4":           "Claude O4",
        "google/gemini-2.0-flash":           "Gemini 2.0F",
        "google/gemini-1.5-pro":             "Gemini 1.5P",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    }
    return MAP.get(model_id, model_id.split("/")[-1])


def _short_pt(pt: str) -> str:
    return {
        "authority":              "Authority",
        "majority":               "Majority",
        "emotional":              "Emotional",
        "iterative_contradiction":"Iterative",
    }.get(pt, pt)


def _safe(v: Optional[float], default: float = 0.0) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


# ── Figure M1: MEBS Score bar chart ──────────────────────────────────────────

def plot_mebs_score(
    model_metrics_rows: list[dict],
    output_path: str | Path,
    bootstrap: Optional[dict] = None,
) -> None:
    """Bar chart of MEBS score per model with optional bootstrap CI error bars."""
    models = sorted(set(r["model"] for r in model_metrics_rows))
    values = [_safe(next((r["mebs_score"] for r in model_metrics_rows if r["model"]==m), 0.0))
              for m in models]
    labels = [_short(m) for m in models]

    yerr = None
    if bootstrap:
        lo = [bootstrap.get(m, {}).get("ci_lower", v) for m, v in zip(models, values)]
        hi = [bootstrap.get(m, {}).get("ci_upper", v) for m, v in zip(models, values)]
        yerr = [[max(0, v-l) for v,l in zip(values,lo)],
                [max(0, h-v) for v,h in zip(values,hi)]]

    fig, ax = plt.subplots(figsize=SIZE_BAR)
    x    = np.arange(len(models))
    bars = ax.bar(x, values, color=PALETTE[:len(models)], edgecolor="white",
                  linewidth=0.8, width=0.55,
                  yerr=yerr, capsize=5,
                  error_kw={"elinewidth":1.5, "ecolor":"#333"})

    # Chance line at 0.5
    ax.axhline(0.5, color="#999", linestyle="--", linewidth=1.2, label="Chance (0.5)")
    ax.legend(fontsize=9, loc="upper right")

    ax.set_xticks(x); ax.set_xticklabels(labels, **FONT_TK)
    ax.set_ylabel("MEBS Score  P(epistemic attr | social cause)", **FONT_L)
    ax.set_title("Figure M1 — Meta-Epistemic Blindspot Score by Model", **FONT_T)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{v:.1%}", ha="center", va="bottom", fontsize=9, color="#333")

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure M1 → %s", output_path)


# ── Figure M2: Attribution confusion matrix per model ────────────────────────

def plot_attribution_confusion(
    trial_rows: list[dict],
    output_path: str | Path,
) -> None:
    """
    For each model: two stacked bars (social cause / epistemic cause)
    showing the distribution of attributions.
    """
    models     = sorted(set(r["model"] for r in trial_rows))
    attr_order = ["epistemic", "social", "mixed", "uncertain"]
    causes     = ["social", "epistemic"]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(3 * n_models + 1, 5),
                              sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        valid = [r for r in trial_rows
                 if r["model"] == model
                 and r.get("attribution") not in ("", "parse_error", None)
                 and r.get("true_cause") in ("social", "epistemic")]

        bottoms = [0.0, 0.0]
        legend_patches = []
        for attr in attr_order:
            heights = []
            for i, cause in enumerate(causes):
                subset = [r for r in valid if r["true_cause"] == cause]
                n = len(subset)
                k = sum(1 for r in subset if r["attribution"] == attr)
                heights.append(k / n if n > 0 else 0.0)

            color = ATTR_COLORS.get(attr, "#aaa")
            ax.bar([0, 1], heights, bottom=bottoms, color=color,
                   edgecolor="white", linewidth=0.5, width=0.6, label=attr)
            legend_patches.append(mpatches.Patch(color=color, label=attr.capitalize()))
            bottoms = [b + h for b, h in zip(bottoms, heights)]

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Social\ncause", "Epistemic\ncause"], fontsize=8)
        ax.set_title(_short(model), fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        if ax == axes[0]:
            ax.set_ylabel("Proportion of attributions", **FONT_L)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Shared legend
    handles = [mpatches.Patch(color=ATTR_COLORS[a], label=a.capitalize())
               for a in attr_order]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Figure M2 — Attribution Distribution by True Cause", **FONT_T, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure M2 → %s", output_path)


# ── Figure M3: SC vs EC scatter ───────────────────────────────────────────────

def plot_coherence_scatter(
    model_metrics_rows: list[dict],
    output_path: str | Path,
) -> None:
    """
    Scatter plot: Social Coherence (x) vs Epistemic Coherence (y).
    Points above the diagonal = epistemic dominates.
    Points below = social dominates (blindspot territory).
    """
    fig, ax = plt.subplots(figsize=SIZE_SCAT)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="#bbb", linestyle="--", linewidth=1.2,
            label="SC = EC (neutral)")

    for idx, row in enumerate(sorted(model_metrics_rows, key=lambda r: r["model"])):
        sc  = _safe(row.get("social_coherence"))
        ec  = _safe(row.get("epistemic_coherence"))
        gap = sc - ec
        color = PALETTE[idx % len(PALETTE)]
        ax.scatter(sc, ec, color=color, s=120, zorder=4)
        # Arrow from diagonal to point
        ax.annotate(
            _short(row["model"]),
            xy=(sc, ec), xytext=(sc + 0.02, ec - 0.04),
            fontsize=8.5, color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )
        # Gap annotation
        ax.annotate(
            f"gap={gap:+.2f}",
            xy=(sc, ec), xytext=(sc + 0.02, ec - 0.07),
            fontsize=7.5, color="#666",
        )

    ax.set_xlabel("Social Coherence  P(answer = false_claim)", **FONT_L)
    ax.set_ylabel("Epistemic Coherence  P(answer = ground_truth)", **FONT_L)
    ax.set_title("Figure M3 — Social vs Epistemic Coherence by Model", **FONT_T)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Shaded regions
    ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.04, color="#E53935",
                    label="Social dominates (SC > EC)")
    ax.fill_between([0, 1], [1, 1], [0, 1], alpha=0.04, color="#43A047",
                    label="Epistemic dominates (EC > SC)")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure M3 → %s", output_path)


# ── Figure M4: Attribution by pressure type heatmap ──────────────────────────

def plot_attribution_pressure_heatmap(
    attribution_by_pressure: list[dict],
    output_path: str | Path,
) -> None:
    """
    Heatmap: P(epistemic attribution) for each (model, pressure_type) cell.
    Social-cause trials only.
    High values = deep blindspot for that pressure type.
    """
    if not attribution_by_pressure:
        logger.warning("No attribution-by-pressure data; skipping Figure M4.")
        return

    models = sorted(set(r["model"] for r in attribution_by_pressure))
    pts    = sorted(set(r["pressure_type"] for r in attribution_by_pressure))

    # Build matrix
    matrix = np.full((len(models), len(pts)), fill_value=np.nan)
    for r in attribution_by_pressure:
        i = models.index(r["model"])
        j = pts.index(r["pressure_type"])
        matrix[i, j] = r.get("p_epistemic", np.nan)
    matrix = np.nan_to_num(matrix, nan=0.0)

    fig, ax = plt.subplots(figsize=SIZE_HEAT)
    im   = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(epistemic attribution | social cause)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    ax.set_xticks(np.arange(len(pts)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([_short_pt(p) for p in pts], **FONT_TK)
    ax.set_yticklabels([_short(m) for m in models], **FONT_TK)
    ax.set_title("Figure M4 — Blindspot Rate by Model × Pressure Type", **FONT_T)

    for i in range(len(models)):
        for j in range(len(pts)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if (v > 0.6 or v < 0.25) else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure M4 → %s", output_path)


# ── Figure M5: Blindspot Specificity vs MEBS Score ───────────────────────────

def plot_blindspot_diagnostic(
    model_metrics_rows: list[dict],
    output_path: str | Path,
) -> None:
    """
    The key diagnostic figure.

    X-axis: MEBS Score (how often the model says 'epistemic' given social cause)
    Y-axis: Blindspot Specificity
            = |P(ep|social) - P(ep|epistemic)|
            High specificity = model CAN distinguish causes (low blindspot)
            Low  specificity = model CANNOT distinguish (deep blindspot)

    Quadrants:
      High MEBS + Low Specificity  → Deep blindspot (can't tell + always says epistemic)
      Low  MEBS + High Specificity → Self-aware (knows when it's being pressured)
      High MEBS + High Specificity → Curious: says epistemic often but with some awareness
      Low  MEBS + Low Specificity  → Rare; model says social even for genuine evidence
    """
    fig, ax = plt.subplots(figsize=SIZE_DIAG)

    # Quadrant shading
    ax.axhline(0.5, color="#ddd", linewidth=0.8)
    ax.axvline(0.5, color="#ddd", linewidth=0.8)
    ax.fill_between([0.5, 1.05], [0, 0],   [0.5, 0.5], alpha=0.06, color="#FB8C00")
    ax.fill_between([0.5, 1.05], [0.5, 0.5],[1.1, 1.1], alpha=0.06, color="#43A047")
    ax.fill_between([-0.05, 0.5],[0, 0],   [1.1, 1.1], alpha=0.04, color="#2196F3")
    ax.fill_between([0.5, 1.05], [0, 0],   [0.5, 0.5], alpha=0.0)

    # Quadrant labels
    ax.text(0.75, 0.18, "Deep\nBlindspot", ha="center", va="center",
            fontsize=8.5, color="#E53935", style="italic", alpha=0.8)
    ax.text(0.75, 0.78, "High MEBS\n+ Self-aware", ha="center", va="center",
            fontsize=8, color="#555", style="italic", alpha=0.8)
    ax.text(0.22, 0.55, "Self-aware\n(low blindspot)", ha="center", va="center",
            fontsize=8.5, color="#43A047", style="italic", alpha=0.8)

    for idx, row in enumerate(sorted(model_metrics_rows, key=lambda r: r["model"])):
        mebs = _safe(row.get("mebs_score"))
        spec = _safe(row.get("blindspot_specificity"))
        color = PALETTE[idx % len(PALETTE)]
        ax.scatter(mebs, spec, color=color, s=130, zorder=5)
        ax.annotate(
            _short(row["model"]),
            xy=(mebs, spec),
            xytext=(mebs + 0.02, spec + 0.02),
            fontsize=9, color=color,
        )

    ax.set_xlabel("MEBS Score  P(epistemic attr | social cause)", **FONT_L)
    ax.set_ylabel("Blindspot Specificity  |P(ep|social) − P(ep|epistemic)|", **FONT_L)
    ax.set_title("Figure M5 — Blindspot Diagnostic: Severity × Specificity", **FONT_T)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure M5 → %s", output_path)


# ── Convenience: generate all MEBS figures ───────────────────────────────────

def generate_all_mebs_figures(
    analysis_results:    dict,
    model_metrics_rows:  list[dict],
    trial_rows:          list[dict],
    output_dir:          str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    boot = analysis_results["statistical_tests"].get("bootstrap_mebs_score", {})
    attr_by_pt  = analysis_results.get("attribution_by_pressure", [])

    plot_mebs_score(
        model_metrics_rows, out / "figM1_mebs_score.png", bootstrap=boot
    )
    plot_attribution_confusion(
        trial_rows, out / "figM2_attribution_confusion.png"
    )
    plot_coherence_scatter(
        model_metrics_rows, out / "figM3_coherence_scatter.png"
    )
    plot_attribution_pressure_heatmap(
        attr_by_pt, out / "figM4_attribution_pressure_heatmap.png"
    )
    plot_blindspot_diagnostic(
        model_metrics_rows, out / "figM5_blindspot_diagnostic.png"
    )

    logger.info("All 5 MEBS figures saved → %s", out)
