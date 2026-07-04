"""Automated, rater-free lexical indicators (a resource for the paper).

These are objective, fully reproducible text measures computed directly from the
model responses -- no human rater, no API key. They corroborate the subjective
human ratings with something that cannot be accused of bias. The lexicons are
small and hand-crafted, so treat these as EXPLORATORY indicators, not validated
instruments (say so in the paper).

Measures (rate = occurrences per 100 words):
  validation  : language that agrees with / amplifies the user's framing
  reframing   : language that introduces another perspective or gently pushes back
  process     : language about effort, control, action, learning, next steps
  outcome     : language fixating on the outcome / catastrophic self-labels
  words       : response length

Outputs
-------
    results/objective_metrics.csv   per-prompt, per-condition raw measures
    results/objective_summary.csv   per-measure paired stats (+ 95% bootstrap CI)
    results/fig_objective.png       grouped bar chart (copy in paper/)
    paper/objective_table.tex       table fragment for the paper
    paper/objective_macros.tex      \newcommand macros for inline numbers

Run:  python scripts/objective_metrics.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Transparent, hand-crafted lexicons (lowercase substring match).
LEXICONS = {
    "validation": [
        "valid", "understandable", "makes sense", "you're right", "you are right",
        "completely", "totally", "absolutely", "of course", "you deserve",
        "no wonder", "justified", "so hard", "awful", "terrible", "genuinely",
        "really sorry", "i'm sorry", "exhausting", "so human", "human reaction",
    ],
    "reframing": [
        "but ", "however", "though", "that said", "on the other hand",
        "push back", "reframe", "notice how", "notice the", "doesn't mean",
        "does not mean", "not the truth", "not a fact", "worth asking",
        "worth examining", "consider", "perspective", "might be worth",
        "is it that", "what if",
    ],
    "process": [
        "effort", "control", "what you can", "within your", "focus on",
        "practice", "step", "next step", "action", "learn", "growth", "improve",
        "work toward", "work on", "prepare", "preparation", "path", "showing up",
        "keep showing", "figure out", "move forward",
    ],
    "outcome": [
        "worthless", "pointless", "failure", "loser", "hopeless", "ruined",
        "never", "doomed", "no point", "all for nothing", "for nothing",
    ],
}
LEX_KEYS = list(LEXICONS)


def word_count(text: str) -> int:
    return max(len(re.findall(r"[a-z']+", text.lower())), 1)


def rate_per_100w(text: str, terms: list[str]) -> float:
    t = text.lower()
    hits = sum(t.count(term) for term in terms)
    return 100.0 * hits / word_count(text)


def boot_ci(diff: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_stats(base: np.ndarray, rag: np.ndarray) -> dict:
    diff = rag - base
    out = {
        "n": int(len(diff)),
        "baseline_mean": float(np.mean(base)),
        "baseline_sd": float(np.std(base, ddof=1)) if len(base) > 1 else float("nan"),
        "nishkama_mean": float(np.mean(rag)),
        "nishkama_sd": float(np.std(rag, ddof=1)) if len(rag) > 1 else float("nan"),
        "mean_diff": float(np.mean(diff)),
        "cohen_dz": float("nan"), "t_p": float("nan"), "wilcoxon_p": float("nan"),
    }
    out["ci_low"], out["ci_high"] = boot_ci(diff)
    if len(diff) > 1 and np.std(diff, ddof=1) > 0:
        out["cohen_dz"] = float(np.mean(diff) / np.std(diff, ddof=1))
        try:
            from scipy import stats
            out["t_p"] = float(stats.ttest_rel(rag, base).pvalue)
            if not np.all(diff == 0):
                out["wilcoxon_p"] = float(stats.wilcoxon(rag, base).pvalue)
        except Exception:  # noqa: BLE001
            pass
    return out


def _fmt(x, dp=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}f}"


def _fmt_p(x):
    if x is None or np.isnan(x):
        return "--"
    return "$<.001$" if x < 0.001 else f"{x:.3f}"


def main() -> int:
    if not config.MODEL_OUTPUTS_CSV.exists():
        print("ERROR: run run_experiment.py first.", file=sys.stderr)
        return 1
    df = pd.read_csv(config.MODEL_OUTPUTS_CSV)
    df = df.sort_values("run").drop_duplicates(subset="id", keep="first")

    # Per-response measures.
    rows = []
    for r in df.itertuples(index=False):
        rec = {"id": r.id, "category": r.category}
        for cond, col in [("baseline", "baseline_response"), ("nishkama", "nishkama_response")]:
            text = getattr(r, col)
            rec[f"{cond}_words"] = word_count(text)
            for k in LEX_KEYS:
                rec[f"{cond}_{k}"] = rate_per_100w(text, LEXICONS[k])
        rows.append(rec)
    per = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per.to_csv(config.RESULTS_DIR / "objective_metrics.csv", index=False)

    measures = LEX_KEYS + ["words"]
    stats_by = {}
    summ_rows = []
    for m in measures:
        s = paired_stats(per[f"baseline_{m}"].to_numpy(float),
                         per[f"nishkama_{m}"].to_numpy(float))
        stats_by[m] = s
        summ_rows.append({"measure": m, **{k: (round(v, 3) if isinstance(v, float) else v)
                                            for k, v in s.items()}})
    pd.DataFrame(summ_rows).to_csv(config.RESULTS_DIR / "objective_summary.csv", index=False)

    # Chart: the four lexical rates, baseline vs RAG.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(LEX_KEYS))
    w = 0.38
    base = [stats_by[m]["baseline_mean"] for m in LEX_KEYS]
    rag = [stats_by[m]["nishkama_mean"] for m in LEX_KEYS]
    base_e = [(stats_by[m]["baseline_sd"] or 0) / np.sqrt(stats_by[m]["n"]) for m in LEX_KEYS]
    rag_e = [(stats_by[m]["nishkama_sd"] or 0) / np.sqrt(stats_by[m]["n"]) for m in LEX_KEYS]
    ax.bar(x - w / 2, base, w, yerr=base_e, capsize=5, label="Baseline", color="#9aa5b1")
    ax.bar(x + w / 2, rag, w, yerr=rag_e, capsize=5, label="Nishkama (RAG)", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels([k.capitalize() for k in LEX_KEYS])
    ax.set_ylabel("Rate per 100 words")
    ax.set_title("Automated lexical indicators by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "fig_objective.png", dpi=200)
    fig.savefig(config.PAPER_DIR / "fig_objective.png", dpi=200)
    plt.close(fig)

    # LaTeX table.
    lines = [
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Indicator & Baseline & Nishkama & $\Delta$ & 95\% CI & $p$ \\",
        r"\midrule",
    ]
    label = {"validation": "Validation", "reframing": "Reframing",
             "process": "Process", "outcome": "Outcome-fixation", "words": "Length (words)"}
    for m in measures:
        s = stats_by[m]
        lines.append(
            f"{label[m]} & {_fmt(s['baseline_mean'])} & {_fmt(s['nishkama_mean'])} & "
            f"{_fmt(s['mean_diff'])} & [{_fmt(s['ci_low'])}, {_fmt(s['ci_high'])}] & "
            f"{_fmt_p(s['t_p'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}",
              "% Rates are per 100 words; Length is raw word count. "
              "Exploratory hand-crafted lexicons."]
    (config.PAPER_DIR / "objective_table.tex").write_text("\n".join(lines) + "\n")

    # Macros.
    cap = {"validation": "Validation", "reframing": "Reframing", "process": "ProcessLex",
           "outcome": "Outcome", "words": "Words"}
    ml = ["% Auto-generated by objective_metrics.py."]
    for m in measures:
        s = stats_by[m]
        c = cap[m]
        ml += [
            f"\\newcommand{{\\Obj{c}Baseline}}{{{_fmt(s['baseline_mean'])}}}",
            f"\\newcommand{{\\Obj{c}Nishkama}}{{{_fmt(s['nishkama_mean'])}}}",
            f"\\newcommand{{\\Obj{c}Diff}}{{{_fmt(s['mean_diff'])}}}",
            f"\\newcommand{{\\Obj{c}P}}{{{_fmt_p(s['t_p'])}}}",
        ]
    (config.PAPER_DIR / "objective_macros.tex").write_text("\n".join(ml) + "\n")

    print("=== Objective lexical indicators (per 100 words) ===")
    print(pd.DataFrame(summ_rows)[["measure", "baseline_mean", "nishkama_mean",
                                   "mean_diff", "ci_low", "ci_high", "t_p"]].to_string(index=False))
    print("\nWrote results/objective_metrics.csv, results/objective_summary.csv")
    print("Wrote results/fig_objective.png (+ paper/), paper/objective_table.tex, paper/objective_macros.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
