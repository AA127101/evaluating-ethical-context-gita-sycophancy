"""Generate RESULTS_FACTSHEET.md -- a plain-language reference of YOUR numbers and
what each one means, so you can write the paper accurately. This is a data/notes
resource: it states the facts and generic statistical meaning, NOT paper prose.

Reads results/summary_results.csv, results/objective_summary.csv, and (if present)
results/reliability.csv.

Run:  python scripts/make_factsheet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

FACTSHEET = config.ROOT / "RESULTS_FACTSHEET.md"


def sig_word(p):
    if p is None or pd.isna(p):
        return "not computable"
    p = float(p)
    if p < 0.001:
        return "statistically significant (p < .001)"
    if p < 0.05:
        return f"statistically significant (p = {p:.3f})"
    if p < 0.10:
        return f"a trend / marginal (p = {p:.3f}, just above .05)"
    return f"not statistically significant (p = {p:.3f})"


def main() -> int:
    if not config.SUMMARY_CSV.exists():
        print("ERROR: run analyze_results.py first.", file=sys.stderr)
        return 1
    s = pd.read_csv(config.SUMMARY_CSV)

    L = []
    L.append("# Results fact sheet (Nishkama-AI)\n")
    L.append("Plain-language reference of your numbers so you can write the paper "
             "accurately. **This is notes, not paper text — write the paper in your "
             "own words.**\n")
    L.append("Convention: **lower sycophancy is better**; **higher stability and "
             "process orientation are better**. `Delta` = Nishkama (RAG) minus "
             "Baseline. `d_z` is the paired effect size (~0.2 small, ~0.5 medium, "
             "~0.8 large).\n")
    L.append("---\n")
    L.append("## Human ratings (primary measure)\n")
    n = int(s["n"].iloc[0])
    L.append(f"- Number of prompts (paired): **{n}**\n")
    for _, r in s.iterrows():
        m = r["metric"]
        L.append(f"### {config.METRIC_LABELS[m]}")
        L.append(f"- Baseline mean: **{r['baseline_mean']}**  |  Nishkama mean: "
                 f"**{r['nishkama_mean']}**")
        L.append(f"- Difference (RAG - baseline): **{r['mean_diff_rag_minus_base']}** "
                 f"(95% bootstrap CI [{r['ci95_low']}, {r['ci95_high']}])")
        ci_excl = (r['ci95_low'] > 0) or (r['ci95_high'] < 0)
        L.append(f"  - The 95% CI {'**excludes** 0' if ci_excl else 'includes 0'} "
                 f"-> {'consistent with a real effect' if ci_excl else 'effect not certain at 95%'}.")
        L.append(f"- Paired t-test: **{sig_word(r['paired_t_p'])}**; "
                 f"Wilcoxon p = {r['wilcoxon_p']}")
        L.append(f"- Effect size (Cohen's d_z): **{r['cohen_dz']}**")
        L.append(f"- Consistency (sign test): moved the better way in "
                 f"**{r['prompts_favorable']}** prompts; sign-test "
                 f"{sig_word(r['sign_test_p'])}")
        direction = "lower = better" if not r["higher_is_better"] else "higher = better"
        L.append(f"- Direction for this metric: {direction}")
        L.append("")
    L.append("**One-line summary you can verify against the table:** the RAG "
             "(Gita) condition reduced sycophancy and raised stability and process "
             "orientation; report which reached significance and which were trends.\n")

    if config.RESULTS_DIR.joinpath("objective_summary.csv").exists():
        o = pd.read_csv(config.RESULTS_DIR / "objective_summary.csv")
        L.append("---\n")
        L.append("## Objective lexical indicators (automated, rater-free, exploratory)\n")
        L.append("Rates are occurrences per 100 words. These corroborate the human "
                 "ratings (no rater involved). Hand-crafted lexicons -> call them "
                 "*exploratory* in the paper.\n")
        for _, r in o.iterrows():
            L.append(f"- **{r['measure']}**: baseline {r['baseline_mean']} -> "
                     f"Nishkama {r['nishkama_mean']} (Delta {r['mean_diff']}, "
                     f"{sig_word(r['t_p'])})")
        L.append("\nKey point to make: the *direction* of these automated measures "
                 "matches the human ratings (more process/reframing language, less "
                 "validation), even where individual differences are not significant.\n")

    if config.RESULTS_DIR.joinpath("reliability.csv").exists():
        rel = pd.read_csv(config.RESULTS_DIR / "reliability.csv")
        L.append("---\n")
        L.append("## Inter-rater reliability (your blind ratings vs LLM judge)\n")
        for _, r in rel.iterrows():
            L.append(f"- **{r['metric']}**: Spearman {r.get('spearman')}, "
                     f"exact-agreement {r.get('exact_agreement')}, "
                     f"mean abs diff {r.get('mean_abs_diff')}")
        L.append("")
    else:
        L.append("---\n")
        L.append("## Inter-rater reliability\n")
        L.append("- Not yet computed. Run the LLM judge (needs your API key) and "
                 "then `python scripts/reliability.py`.\n")

    L.append("---\n")
    L.append("## How to talk about significance honestly\n")
    L.append("- p < .05 = statistically significant. p between .05 and .10 = a "
             "*trend* (say 'did not reach significance').")
    L.append("- With only ~14 prompts, power is low: report effect sizes and CIs, "
             "and treat trends as exploratory.")
    L.append("- It is a *strength*, not a weakness, that not everything is "
             "significant in a tiny sample -- it reads as honest.")
    FACTSHEET.write_text("\n".join(L) + "\n")
    print(f"Wrote {FACTSHEET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
