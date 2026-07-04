"""Generate three additional figures for the paper (resources, not prose):

  fig_methodology.png  - schematic of the Baseline vs Nishkama (RAG) pipeline
  fig_forest.png       - mean paired differences with 95% bootstrap CIs
  fig_perprompt.png    - per-prompt RAG-minus-baseline differences (consistency)

Reads results/summary_results.csv and results/per_prompt_diffs.csv (from
analyze_results.py). Copies all figures into paper/ as well.

Run:  python scripts/extra_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

BASE_COLOR, RAG_COLOR, GOOD, BAD = "#9aa5b1", "#e07a5f", "#5b8c5a", "#c0c0c0"


def save(fig, name):
    fig.savefig(config.RESULTS_DIR / name, dpi=200, bbox_inches="tight")
    fig.savefig(config.PAPER_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def methodology_figure():
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.5,rounding_size=2.2",
                                    linewidth=1.4, edgecolor="#3a3a3a", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=17, linewidth=1.5, color="#3a3a3a",
                                     shrinkA=2, shrinkB=2))

    # Shared prompt (left)
    box(3, 45, 21, 15, "Emotionally\ncharged prompt", "#eef2f6")
    # Baseline branch (top row)
    box(35, 71, 22, 14, "LLM\n(same model)", "#eef2f6")
    box(67, 71, 29, 14, "Baseline\nresponse", BASE_COLOR)
    # Nishkama branch (retrieve -> LLM -> response)
    box(33, 44, 26, 16, "Retrieve top-2\nGita verses\n(28-verse set)", "#fbeee9")
    box(35, 20, 22, 14, "LLM\n(same model)", "#eef2f6")
    box(67, 20, 23, 14, "Nishkama (RAG)\nresponse", RAG_COLOR, fs=9.5)
    # Scoring (wide, bottom)
    box(15, 3, 81, 12, "Blind human ratings  +  LLM judge  +  lexical metrics\n"
        "(sycophancy / stability / process)", "#f3f0e9", fs=10)

    # Prompt branches
    arrow(24, 55, 35, 76)      # prompt -> baseline LLM (up-right)
    arrow(24, 50, 33, 51)      # prompt -> retrieve
    arrow(57, 78, 67, 78)      # baseline LLM -> baseline response
    arrow(46, 44, 46, 34)      # retrieve -> nishkama LLM (down)
    arrow(57, 27, 67, 27)      # nishkama LLM -> nishkama response
    # Responses -> scoring (no box crossings)
    arrow(78, 20, 78, 15)      # nishkama response straight down to scoring
    arrow(93, 71, 93, 15)      # baseline response down the right margin (clears Nishkama at x<=90)

    ax.text(13, 38, "same prompt\n+ verses", ha="center", fontsize=8, color="#888")
    ax.text(50, 95, "Only difference between conditions: the retrieved verses",
            ha="center", fontsize=11, style="italic", color="#333")
    fig.tight_layout()
    save(fig, "fig_methodology.png")


# ---------------------------------------------------------------------------
def forest_figure():
    s = pd.read_csv(config.SUMMARY_CSV)
    metrics = list(s["metric"])
    y = np.arange(len(metrics))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for yi, (_, r) in zip(y, s.iterrows()):
        lo, hi, md = r["ci95_low"], r["ci95_high"], r["mean_diff_rag_minus_base"]
        favorable = (md > 0) if r["higher_is_better"] else (md < 0)
        c = GOOD if favorable else "#b04a3a"
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=3.2, solid_capstyle="round", zorder=2)
        ax.plot(md, yi, "o", color=c, markersize=12, zorder=3,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.annotate(f"{md:+.2f}", (md, yi), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=10, color="#333")
    ax.axvline(0, color="#999", linestyle="--", linewidth=1.3, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([config.METRIC_LABELS[m] for m in metrics], fontsize=11.5)
    ax.set_ylim(-0.7, len(metrics) - 0.3)
    ax.margins(x=0.10)
    ax.set_xlabel("Mean paired difference (Nishkama − Baseline), with 95% CI", fontsize=10.5)
    ax.set_title("Effect of Gita context by metric", fontsize=13.5, pad=24)
    ax.text(0.5, 1.03, "point = mean difference   ·   bar = 95% CI   ·   dashed line = no effect (0)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#777")
    fig.text(0.5, 0.01,
             "Sycophancy: more negative is better.    Stability & process: more positive is better.",
             ha="center", va="bottom", fontsize=9, color="#666")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "fig_forest.png")


# ---------------------------------------------------------------------------
def perprompt_figure():
    p = pd.read_csv(config.RESULTS_DIR / "per_prompt_diffs.csv")
    diff_cols = {m: f"{m}_diff(rag-base)" for m in config.METRICS}
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    ids = p["id"].tolist()
    yy = np.arange(len(ids))[::-1]
    for ax, m in zip(axes, config.METRICS):
        vals = p[diff_cols[m]].to_numpy(float)
        hb = config.METRIC_HIGHER_IS_BETTER[m]
        colors = [GOOD if ((v > 0) if hb else (v < 0)) else (BAD if v == 0 else "#b04a3a")
                  for v in vals]
        ax.barh(yy, vals, color=colors)
        ax.axvline(0, color="#888", linewidth=1)
        ax.set_title(config.METRIC_LABELS[m] +
                     ("\n(neg = better)" if not hb else "\n(pos = better)"), fontsize=9)
        ax.set_xlabel("RAG - Baseline")
    axes[0].set_yticks(yy); axes[0].set_yticklabels(ids, fontsize=7)
    fig.suptitle("Per-prompt differences (consistency across the 14 prompts)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig_perprompt.png")


def main() -> int:
    if not config.SUMMARY_CSV.exists():
        print("ERROR: run analyze_results.py first.", file=sys.stderr)
        return 1
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    methodology_figure()
    forest_figure()
    perprompt_figure()
    print("Wrote fig_methodology.png, fig_forest.png, fig_perprompt.png "
          "(results/ and paper/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
