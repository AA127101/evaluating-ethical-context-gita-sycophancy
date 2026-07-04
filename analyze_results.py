"""Analyse scored results: statistics, charts, and LaTeX fragments (resources).

Accepts either
  * a BLIND scoring sheet (results/scoring_sheet.csv with A_*/B_* columns) plus
    the key (results/blind_key.csv), or
  * a LABELLED scoring file with baseline_*/nishkama_* columns (e.g. the LLM
    judge output, or a sheet you filled in directly).

Outputs
-------
    results/summary_results.csv     per-metric means, diffs, 95% bootstrap CI,
                                    paired t / Wilcoxon, Cohen's d_z, sign test
    results/per_prompt_diffs.csv    per-prompt RAG-minus-baseline differences
    results/fig_sycophancy.png      bar charts (copies also placed in paper/)
    results/fig_stability.png
    results/fig_process.png
    results/fig_overview.png
    paper/results_table.tex         results table (\input by main.tex)
    paper/results_macros.tex        \newcommand macros for inline numbers

Run:
    python analyze_results.py
    python analyze_results.py --scores results/llm_scoring_sheet.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Load + un-blind
# ---------------------------------------------------------------------------
def load_scores(scores_path: Path, key_path: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_path)
    cols = set(df.columns)
    labelled = all(f"{c}_{m}" in cols for c in config.CONDITIONS for m in config.METRICS)
    if labelled:
        return df
    blind = all(f"{ab}_{m}" in cols for ab in ("A", "B") for m in config.METRICS)
    if not blind:
        raise ValueError("Need baseline_*/nishkama_* or A_*/B_* columns.")
    if not key_path.exists():
        raise FileNotFoundError(f"Blind sheet detected but key {key_path} is missing.")
    key = pd.read_csv(key_path).set_index("id")
    rows = []
    for row in df.itertuples(index=False):
        rec = {"id": row.id, "category": getattr(row, "category", "")}
        a_cond = key.loc[row.id, "A_condition"]
        b_cond = key.loc[row.id, "B_condition"]
        for m in config.METRICS:
            rec[f"{a_cond}_{m}"] = getattr(row, f"A_{m}")
            rec[f"{b_cond}_{m}"] = getattr(row, f"B_{m}")
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def boot_ci(diff: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return tuple(float(v) for v in np.percentile(means, [2.5, 97.5]))


def paired_stats(baseline: np.ndarray, nishkama: np.ndarray, higher_better: bool) -> dict:
    mask = ~(np.isnan(baseline) | np.isnan(nishkama))
    b, n = baseline[mask], nishkama[mask]
    diff = n - b
    out = {
        "n": int(mask.sum()),
        "baseline_mean": float(np.mean(b)) if len(b) else float("nan"),
        "baseline_sd": float(np.std(b, ddof=1)) if len(b) > 1 else float("nan"),
        "nishkama_mean": float(np.mean(n)) if len(n) else float("nan"),
        "nishkama_sd": float(np.std(n, ddof=1)) if len(n) > 1 else float("nan"),
        "mean_diff": float(np.mean(diff)) if len(diff) else float("nan"),
        "ci_low": float("nan"), "ci_high": float("nan"),
        "t_p": float("nan"), "wilcoxon_p": float("nan"), "cohen_dz": float("nan"),
        "n_favorable": 0, "n_nonzero": 0, "sign_p": float("nan"),
    }
    out["ci_low"], out["ci_high"] = boot_ci(diff)
    # Sign test: how many prompts moved in the predicted (better) direction.
    favorable_mask = (diff > 0) if higher_better else (diff < 0)
    nonzero = diff != 0
    out["n_favorable"] = int(np.sum(favorable_mask & nonzero))
    out["n_nonzero"] = int(np.sum(nonzero))
    if len(diff) > 1 and np.std(diff, ddof=1) > 0:
        out["cohen_dz"] = float(np.mean(diff) / np.std(diff, ddof=1))
        try:
            from scipy import stats
            out["t_p"] = float(stats.ttest_rel(n, b).pvalue)
            if not np.all(diff == 0):
                out["wilcoxon_p"] = float(stats.wilcoxon(n, b).pvalue)
            if out["n_nonzero"] > 0:
                out["sign_p"] = float(stats.binomtest(
                    out["n_favorable"], out["n_nonzero"], 0.5,
                    alternative="greater").pvalue)
        except Exception:  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
BASE_COLOR, RAG_COLOR = "#9aa5b1", "#e07a5f"


def _sem(sd, n):
    return sd / np.sqrt(n) if n and n > 0 and not np.isnan(sd) else 0.0


def bar_chart(metric, s, path):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    means = [s["baseline_mean"], s["nishkama_mean"]]
    errs = [_sem(s["baseline_sd"], s["n"]), _sem(s["nishkama_sd"], s["n"])]
    ax.bar([0, 1], means, yerr=errs, capsize=6, color=[BASE_COLOR, RAG_COLOR], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels([config.CONDITION_LABELS[c] for c in config.CONDITIONS])
    ax.set_ylim(0, 5.4); ax.set_ylabel(f"{config.METRIC_LABELS[metric]} (1-5)")
    d = "lower is better" if not config.METRIC_HIGHER_IS_BETTER[metric] else "higher is better"
    ax.set_title(f"{config.METRIC_LABELS[metric]}\n({d})")
    for x, m in zip([0, 1], means):
        if not np.isnan(m):
            ax.text(x, m + 0.12, f"{m:.2f}", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); fig.savefig(config.PAPER_DIR / path.name, dpi=200)
    plt.close(fig)


def overview_chart(stats_by_metric, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    metrics = config.METRICS; x = np.arange(len(metrics)); w = 0.38
    base = [stats_by_metric[m]["baseline_mean"] for m in metrics]
    rag = [stats_by_metric[m]["nishkama_mean"] for m in metrics]
    be = [_sem(stats_by_metric[m]["baseline_sd"], stats_by_metric[m]["n"]) for m in metrics]
    re_ = [_sem(stats_by_metric[m]["nishkama_sd"], stats_by_metric[m]["n"]) for m in metrics]
    ax.bar(x - w / 2, base, w, yerr=be, capsize=5, label="Baseline", color=BASE_COLOR)
    ax.bar(x + w / 2, rag, w, yerr=re_, capsize=5, label="Nishkama (RAG)", color=RAG_COLOR)
    ax.set_xticks(x); ax.set_xticklabels([config.METRIC_LABELS[m] for m in metrics])
    ax.set_ylim(0, 5.4); ax.set_ylabel("Mean rating (1-5)")
    ax.set_title("Baseline vs Nishkama (RAG) across metrics"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); fig.savefig(config.PAPER_DIR / path.name, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def _fmt(x, dp=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}f}"


def _fmt_p(x):
    if x is None or np.isnan(x):
        return "--"
    return "$<.001$" if x < 0.001 else f"{x:.3f}"


def write_latex(stats_by_metric):
    config.PAPER_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Metric & Baseline & Nishkama & $\Delta$ & 95\% CI & $p$ & $d_z$ \\",
        r"\midrule",
    ]
    for m in config.METRICS:
        s = stats_by_metric[m]
        lines.append(
            f"{config.METRIC_LABELS[m]} & "
            f"{_fmt(s['baseline_mean'])} ({_fmt(s['baseline_sd'])}) & "
            f"{_fmt(s['nishkama_mean'])} ({_fmt(s['nishkama_sd'])}) & "
            f"{_fmt(s['mean_diff'])} & [{_fmt(s['ci_low'])}, {_fmt(s['ci_high'])}] & "
            f"{_fmt_p(s['t_p'])} & {_fmt(s['cohen_dz'])} \\\\"
        )
    n = stats_by_metric[config.METRICS[0]]["n"]
    lines += [r"\bottomrule", r"\end{tabular}",
              f"% n = {n} prompts. Cells: mean (SD). CI = 95% bootstrap CI of the "
              "paired difference. Lower sycophancy is better; higher stability/process better."]
    (config.PAPER_DIR / "results_table.tex").write_text("\n".join(lines) + "\n")

    cap = {"sycophancy": "Sycophancy", "stability": "Stability", "process": "Process"}
    ml = ["% Auto-generated by analyze_results.py. Do not edit by hand.",
          f"\\newcommand{{\\NPrompts}}{{{n}}}"]
    for m in config.METRICS:
        s = stats_by_metric[m]; c = cap[m]
        ml += [
            f"\\newcommand{{\\MeanBaseline{c}}}{{{_fmt(s['baseline_mean'])}}}",
            f"\\newcommand{{\\MeanNishkama{c}}}{{{_fmt(s['nishkama_mean'])}}}",
            f"\\newcommand{{\\Diff{c}}}{{{_fmt(s['mean_diff'])}}}",
            f"\\newcommand{{\\AbsDiff{c}}}{{{_fmt(abs(s['mean_diff']))}}}",
            f"\\newcommand{{\\CILow{c}}}{{{_fmt(s['ci_low'])}}}",
            f"\\newcommand{{\\CIHigh{c}}}{{{_fmt(s['ci_high'])}}}",
            f"\\newcommand{{\\PValue{c}}}{{{_fmt_p(s['t_p'])}}}",
            f"\\newcommand{{\\CohenD{c}}}{{{_fmt(s['cohen_dz'])}}}",
            f"\\newcommand{{\\Favorable{c}}}{{{s['n_favorable']}}}",
            f"\\newcommand{{\\NonZero{c}}}{{{s['n_nonzero']}}}",
        ]
    (config.PAPER_DIR / "results_macros.tex").write_text("\n".join(ml) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse scored results.")
    parser.add_argument("--scores", default=str(config.SCORING_SHEET_CSV))
    parser.add_argument("--key", default=str(config.BLIND_KEY_CSV))
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        print(f"ERROR: {scores_path} not found.\n"
              "Build it: python scripts/make_scoring_template.py", file=sys.stderr)
        return 1

    df = load_scores(scores_path, Path(args.key))
    score_cols = [f"{c}_{m}" for c in config.CONDITIONS for m in config.METRICS]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[score_cols].isna().all().all():
        print("ERROR: no numeric scores found. Did you fill in the sheet?", file=sys.stderr)
        return 1

    stats_by_metric, summary_rows, per_prompt = {}, [], df[["id"]].copy()
    if "category" in df.columns:
        per_prompt["category"] = df["category"].values
    for m in config.METRICS:
        hb = config.METRIC_HIGHER_IS_BETTER[m]
        s = paired_stats(df[f"baseline_{m}"].to_numpy(float),
                         df[f"nishkama_{m}"].to_numpy(float), hb)
        stats_by_metric[m] = s
        per_prompt[f"{m}_diff(rag-base)"] = (df[f"nishkama_{m}"] - df[f"baseline_{m}"]).values
        improvement = s["mean_diff"] if hb else -s["mean_diff"]
        summary_rows.append({
            "metric": m, "n": s["n"],
            "baseline_mean": round(s["baseline_mean"], 3),
            "nishkama_mean": round(s["nishkama_mean"], 3),
            "mean_diff_rag_minus_base": round(s["mean_diff"], 3),
            "improvement_sign_adj": round(improvement, 3),
            "ci95_low": round(s["ci_low"], 3), "ci95_high": round(s["ci_high"], 3),
            "higher_is_better": hb,
            "paired_t_p": round(s["t_p"], 4) if not np.isnan(s["t_p"]) else None,
            "wilcoxon_p": round(s["wilcoxon_p"], 4) if not np.isnan(s["wilcoxon_p"]) else None,
            "cohen_dz": round(s["cohen_dz"], 3) if not np.isnan(s["cohen_dz"]) else None,
            "prompts_favorable": f"{s['n_favorable']}/{s['n_nonzero']}",
            "sign_test_p": round(s["sign_p"], 4) if not np.isnan(s["sign_p"]) else None,
        })

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.SUMMARY_CSV, index=False)
    per_prompt.to_csv(config.RESULTS_DIR / "per_prompt_diffs.csv", index=False)

    bar_chart("sycophancy", stats_by_metric["sycophancy"], config.RESULTS_DIR / "fig_sycophancy.png")
    bar_chart("stability", stats_by_metric["stability"], config.RESULTS_DIR / "fig_stability.png")
    bar_chart("process", stats_by_metric["process"], config.RESULTS_DIR / "fig_process.png")
    overview_chart(stats_by_metric, config.RESULTS_DIR / "fig_overview.png")
    write_latex(stats_by_metric)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {config.SUMMARY_CSV}")
    print(f"Wrote {config.RESULTS_DIR / 'per_prompt_diffs.csv'}")
    print("Wrote figures -> results/fig_*.png (copies in paper/)")
    print("Wrote LaTeX   -> paper/results_table.tex, paper/results_macros.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
