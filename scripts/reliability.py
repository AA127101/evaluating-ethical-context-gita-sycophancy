"""Inter-rater reliability: how well do your blind human ratings agree with the
LLM judge? (A resource for the paper -- you interpret it.)

For each metric it pools all responses (14 prompts x 2 conditions = 28) and
compares the two raters with:
  * Spearman rank correlation (monotonic agreement)
  * Pearson correlation
  * exact-agreement rate (identical 1-5 score)
  * mean absolute difference

Needs:
  results/scoring_sheet.csv + results/blind_key.csv   (your human ratings)
  results/llm_scoring_sheet.csv                        (from scripts/llm_judge.py)

Output -> results/reliability.csv   (+ printed table)

Run:  python scripts/reliability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def human_long(sheet: Path, key: Path) -> pd.DataFrame:
    df = pd.read_csv(sheet)
    k = pd.read_csv(key).set_index("id")
    rows = []
    for r in df.itertuples(index=False):
        for ab in ("A", "B"):
            cond = k.loc[r.id, f"{ab}_condition"]
            rec = {"id": r.id, "condition": cond}
            for m in config.METRICS:
                rec[m] = getattr(r, f"{ab}_{m}")
            rows.append(rec)
    return pd.DataFrame(rows)


def judge_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for r in df.itertuples(index=False):
        for cond in config.CONDITIONS:
            rec = {"id": r.id, "condition": cond}
            for m in config.METRICS:
                rec[m] = getattr(r, f"{cond}_{m}")
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    for p in (config.SCORING_SHEET_CSV, config.BLIND_KEY_CSV):
        if not Path(p).exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1
    if not Path(config.LLM_SCORING_CSV).exists():
        print(f"ERROR: {config.LLM_SCORING_CSV} not found.\n"
              "Run the LLM judge first:  python scripts/llm_judge.py", file=sys.stderr)
        return 1

    h = human_long(config.SCORING_SHEET_CSV, config.BLIND_KEY_CSV)
    j = judge_long(config.LLM_SCORING_CSV)
    merged = h.merge(j, on=["id", "condition"], suffixes=("_human", "_judge"))

    try:
        from scipy import stats
        have_scipy = True
    except Exception:  # noqa: BLE001
        have_scipy = False

    rows = []
    for m in config.METRICS:
        a = merged[f"{m}_human"].to_numpy(float)
        b = merged[f"{m}_judge"].to_numpy(float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        rec = {"metric": m, "n_responses": int(len(a)),
               "exact_agreement": round(float(np.mean(a == b)), 3) if len(a) else None,
               "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 3) if len(a) else None,
               "spearman": None, "spearman_p": None, "pearson": None}
        if have_scipy and len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            sr = stats.spearmanr(a, b)
            rec["spearman"] = round(float(sr.correlation), 3)
            rec["spearman_p"] = round(float(sr.pvalue), 4)
            rec["pearson"] = round(float(stats.pearsonr(a, b)[0]), 3)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / "reliability.csv", index=False)
    print("=== Inter-rater reliability: human (blind) vs LLM judge ===")
    print(out.to_string(index=False))
    print(f"\nWrote {config.RESULTS_DIR / 'reliability.csv'}")
    print("Interpretation hint: Spearman > ~0.5 = moderate agreement, > ~0.7 = strong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
