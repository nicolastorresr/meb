"""
visualization.py
Publication-quality figures for the EPP experiment.

Figure 1  — Stability by model (bar + bootstrap CI)
Figure 2  — Compliance by model (bar)
Figure 3  — Confidence Drift by model (boxplot)
Figure 4  — Pressure susceptibility heatmap (compliance)
Figure 5  — EPP Radar (6-dimensional: Stability, Compliance, Confidence Drift,
                        Flexibility, Recovery, Self-Awareness)
Figure 6  — Version Stability: intra- vs inter-family EPP distance
"""

import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np

from metrics import EPP_DIMENSIONS

logger = logging.getLogger(__name__)

DPI           = 300
SIZE_BAR      = (8, 5)
SIZE_BOX      = (8, 5)
SIZE_HEAT     = (10, 6)
SIZE_RADAR    = (7, 7)
SIZE_VS       = (8, 5)

PALETTE = ["#2196F3","#FF5722","#4CAF50","#9C27B0","#FF9800","#00BCD4","#E91E63","#607D8B"]

FONT_T  = {"fontsize": 14, "fontweight": "bold"}
FONT_L  = {"fontsize": 11}
FONT_TK = {"fontsize": 9}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "grid.linestyle": "--",
})

DIM_LABELS = {
    "stability":             "Stability",
    "compliance":            "Compliance",
    "confidence_drift_mean": "Confidence\nDrift",
    "flexibility":           "Flexibility",
    "recovery":              "Recovery",
    "self_awareness":        "Self-\nAwareness",
}


def _short(model_id: str) -> str:
    MAP = {
        "openai/gpt-4o":                        "GPT-4o",
        "openai/gpt-4.1":                       "GPT-4.1",
        "openai/gpt-4.5-preview":               "GPT-4.5",
        "anthropic/claude-sonnet-4-5":          "Claude\nSonnet 4.5",
        "anthropic/claude-sonnet-4.5":          "Claude\nSonnet 4.5",
        "anthropic/claude-opus-4":              "Claude\nOpus 4",
        "google/gemini-2.0-flash":              "Gemini\n2.0 Flash",
        "google/gemini-1.5-pro":                "Gemini\n1.5 Pro",
        "meta-llama/llama-3.3-70b-instruct":    "Llama 3.3\n70B",
        "meta-llama/llama-3.1-405b-instruct":   "Llama 3.1\n405B",
    }
    return MAP.get(model_id, model_id.split("/")[-1])


def _short_pt(pt: str) -> str:
    return {
        "authority":              "Authority",
        "majority":               "Majority",
        "emotional":              "Emotional",
        "iterative_contradiction":"Iterative\nContra.",
    }.get(pt, pt)


# ─── Figure 1: Stability bar ─────────────────────────────────────────────────

def plot_stability_by_model(
    model_metrics: dict[str, dict],
    output_path: str | Path,
    bootstrap_ci: Optional[dict] = None,
) -> None:
    models = sorted(model_metrics)
    values = [_safe(model_metrics[m].get("stability")) for m in models]
    labels = [_short(m) for m in models]

    yerr = None
    if bootstrap_ci:
        lo = [bootstrap_ci.get(m, {}).get("ci_lower", v) or v for m, v in zip(models, values)]
        hi = [bootstrap_ci.get(m, {}).get("ci_upper", v) or v for m, v in zip(models, values)]
        yerr = [[max(0, v-l) for v,l in zip(values,lo)],
                [max(0, h-v) for v,h in zip(values,hi)]]

    fig, ax = plt.subplots(figsize=SIZE_BAR)
    x    = np.arange(len(models))
    bars = ax.bar(x, values, color=PALETTE[:len(models)], edgecolor="white",
                  linewidth=0.8, width=0.55,
                  yerr=yerr, capsize=5,
                  error_kw={"elinewidth":1.5, "ecolor":"#333"})
    ax.set_xticks(x); ax.set_xticklabels(labels, **FONT_TK)
    ax.set_ylabel("Stability (proportion)", **FONT_L)
    ax.set_title("Figure 1 — Epistemic Stability by Model", **FONT_T)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                f"{v:.1%}", ha="center", va="bottom", fontsize=9, color="#333")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure 1 → %s", output_path)


# ─── Figure 2: Compliance bar ────────────────────────────────────────────────

def plot_compliance_by_model(model_metrics: dict, output_path: str | Path) -> None:
    models = sorted(model_metrics)
    values = [_safe(model_metrics[m].get("compliance")) for m in models]
    labels = [_short(m) for m in models]

    fig, ax = plt.subplots(figsize=SIZE_BAR)
    x    = np.arange(len(models))
    bars = ax.bar(x, values, color=PALETTE[:len(models)], edgecolor="white",
                  linewidth=0.8, width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(labels, **FONT_TK)
    ax.set_ylabel("Compliance (proportion)", **FONT_L)
    ax.set_title("Figure 2 — Epistemic Compliance by Model", **FONT_T)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                f"{v:.1%}", ha="center", va="bottom", fontsize=9, color="#333")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure 2 → %s", output_path)


# ─── Figure 3: Confidence Drift boxplot ──────────────────────────────────────

def plot_confidence_drift_boxplot(rows: list[dict], output_path: str | Path) -> None:
    from collections import defaultdict
    models = sorted(set(r["model"] for r in rows))

    def drifts_for(model_rows):
        by_trial = defaultdict(list)
        for r in model_rows:
            by_trial[r["experiment_id"]].append(r)
        out = []
        for turns in by_trial.values():
            t0 = next((t for t in turns if t["turn"]==0), None)
            t3 = next((t for t in turns if t["turn"]==3), None)
            if t0 and t3:
                out.append(abs(int(t3["confidence"]) - int(t0["confidence"])))
        return out or [0]

    data   = [drifts_for([r for r in rows if r["model"]==m]) for m in models]
    labels = [_short(m) for m in models]

    fig, ax = plt.subplots(figsize=SIZE_BOX)
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color":"white","linewidth":2},
                    whiskerprops={"linewidth":1.5},
                    capprops={"linewidth":1.5},
                    flierprops={"marker":"o","markersize":4,"alpha":0.5})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticklabels(labels, **FONT_TK)
    ax.set_ylabel("Confidence Drift |Δ| (points)", **FONT_L)
    ax.set_title("Figure 3 — Confidence Drift Distribution by Model", **FONT_T)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure 3 → %s", output_path)


# ─── Figure 4: Pressure heatmap ──────────────────────────────────────────────

def plot_pressure_heatmap(heatmap_data: dict, output_path: str | Path) -> None:
    models = sorted(heatmap_data)
    if not models:
        return
    pts    = sorted(next(iter(heatmap_data.values())))
    matrix = np.array(
        [[_safe(heatmap_data[m].get(pt)) for pt in pts] for m in models],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=SIZE_HEAT)
    im   = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Compliance Rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_xticks(np.arange(len(pts)));    ax.set_xticklabels([_short_pt(p) for p in pts], **FONT_TK)
    ax.set_yticks(np.arange(len(models))); ax.set_yticklabels([_short(m) for m in models], **FONT_TK)
    ax.set_title("Figure 4 — Pressure Susceptibility Heatmap (Compliance)", **FONT_T)
    for i in range(len(models)):
        for j in range(len(pts)):
            v = matrix[i,j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=10, color="white" if v>0.6 else "black", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure 4 → %s", output_path)


# ─── Figure 5: 6-dim EPP Radar ───────────────────────────────────────────────

def plot_epp_radar(
    normalized_profiles: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    dims   = EPP_DIMENSIONS
    labels = [DIM_LABELS[d] for d in dims]
    models = sorted(normalized_profiles)
    n_dims = len(dims)

    angles = np.linspace(0, 2*np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=SIZE_RADAR, subplot_kw={"projection":"polar"})
    handles = []

    for idx, model in enumerate(models):
        color  = PALETTE[idx % len(PALETTE)]
        profile = normalized_profiles[model]
        values  = [_safe(profile.get(d)) for d in dims] + [_safe(profile.get(dims[0]))]
        ax.plot(angles, values, color=color, linewidth=2.0)
        ax.fill(angles, values, color=color, alpha=0.12)
        handles.append(mpatches.Patch(color=color,
                                       label=_short(model).replace("\n"," ")))

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.0"], fontsize=8, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title("Figure 5 — Epistemic Personality Profiles (6D)", pad=20, **FONT_T)
    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.45, 1.15), fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure 5 → %s", output_path)


# ─── Figure 6: Version Stability ─────────────────────────────────────────────

def plot_version_stability(
    vs_result: dict,
    output_path: str | Path,
) -> None:
    """
    Strip-plot + mean markers comparing intra-family vs inter-family
    EPP Euclidean distances.
    """
    pairs = vs_result.get("pair_details", [])
    if not pairs:
        logger.warning("No pair data for Figure 6; skipping.")
        return

    intra = [p["euclidean_distance"] for p in pairs if p["same_family"]]
    inter = [p["euclidean_distance"] for p in pairs if not p["same_family"]]

    fig, ax = plt.subplots(figsize=SIZE_VS)
    jitter = 0.08

    rng = np.random.default_rng(0)
    if intra:
        x_i = rng.uniform(-jitter, jitter, len(intra))
        ax.scatter(x_i, intra, color=PALETTE[0], alpha=0.7, s=60, zorder=3,
                   label="Intra-family")
        ax.plot([-0.25, 0.25], [np.mean(intra)]*2, color=PALETTE[0],
                linewidth=2.5, zorder=4)
    if inter:
        x_e = 1 + rng.uniform(-jitter, jitter, len(inter))
        ax.scatter(x_e, inter, color=PALETTE[1], alpha=0.7, s=60, zorder=3,
                   label="Inter-family")
        ax.plot([0.75, 1.25], [np.mean(inter)]*2, color=PALETTE[1],
                linewidth=2.5, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Intra-family\n(same provider)", "Inter-family\n(different providers)"],
                       fontsize=11)
    ax.set_ylabel("EPP Euclidean Distance (normalized)", **FONT_L)
    ax.set_title("Figure 6 — Version Stability: EPP Distance by Model Family", **FONT_T)
    ax.legend(fontsize=9)

    # Annotate p-value if available
    mw = vs_result.get("mann_whitney_u", {})
    if "p_value" in mw:
        p = mw["p_value"]
        supported = mw.get("hypothesis_supported", False)
        symbol = "✓" if supported else "✗"
        ax.text(0.5, 0.95, f"Mann-Whitney U  p = {p:.3f}  {symbol}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color="#333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="#aaa", alpha=0.9))

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved Figure 6 → %s", output_path)


# ─── convenience: generate all ───────────────────────────────────────────────

def generate_all_figures(
    analysis_results: dict,
    raw_rows: list[dict],
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mm   = analysis_results["model_metrics"]
    hm   = analysis_results["heatmap_data"]
    norm = analysis_results["normalized_profiles"]
    boot = analysis_results["statistical_tests"].get("bootstrap_stability", {})
    vs   = analysis_results["statistical_tests"].get("version_stability", {})

    plot_stability_by_model(mm,   out/"fig1_stability_by_model.png",       boot)
    plot_compliance_by_model(mm,  out/"fig2_compliance_by_model.png")
    plot_confidence_drift_boxplot(raw_rows, out/"fig3_confidence_drift_boxplot.png")
    plot_pressure_heatmap(hm,     out/"fig4_pressure_heatmap.png")
    plot_epp_radar(norm,          out/"fig5_epp_radar_6dim.png")
    plot_version_stability(vs,    out/"fig6_version_stability.png")

    logger.info("All 6 figures saved → %s", out)


def _safe(v: Optional[float], default: float = 0.0) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)
