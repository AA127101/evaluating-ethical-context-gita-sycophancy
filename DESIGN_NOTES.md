# Design notes (Nishkama-AI)

Technical notes on the study design and the codebase. **These are developer
notes / reminders — not the paper.** Use them while you write the paper in your
own words.

## Research question
Does retrieving Bhagavad Gita verses into the prompt change how an LLM responds
to emotionally charged messages, compared with an identical no-context baseline?

## Key design decisions (and why)
- **Clean manipulation.** The system prompt is identical in both conditions and
  never tells the model to be less sycophantic (`config.BASE_SYSTEM`). The only
  thing that differs is whether retrieved verses are prepended. So any measured
  difference is attributable to the verses, not to an instruction.
- **Same model both conditions** (`claude-opus-4-8`) with the same decoding
  settings — removes model/config as a confound.
- **Blind scoring.** Responses are shuffled and anonymized (A/B) before rating;
  the condition labels are revealed only at analysis time
  (`make_scoring_template.py` writes `blind_key.csv`). Reduces rater bias.
- **Scoring direction (standardized).** Sycophancy: higher = more sycophantic =
  worse. Stability and process orientation: higher = better. The analysis
  handles direction automatically.
- **Retrieval.** Top-2 verses by semantic similarity
  (`all-MiniLM-L6-v2`), with TF-IDF and lexical fallbacks. The method used is
  recorded in `model_outputs.csv`.
- **Multiple measures.** Primary = blind human ratings; secondary = optional LLM
  judge; tertiary = automated lexical indicators (rater-free). Convergence
  across measures is the evidence, not any single one.

## File map
- `data/` — `prompts.csv` (14), `gita_dataset.csv` (28 paraphrased verses).
- `run_experiment.py` — generates baseline + RAG responses (resumable, retries).
- `scripts/make_scoring_template.py` — builds the blind scoring sheet + key.
- `analyze_results.py` — paired t/Wilcoxon, Cohen's d_z, 95% bootstrap CIs,
  sign test; figures; LaTeX tables/macros.
- `scripts/objective_metrics.py` — automated lexical indicators.
- `scripts/extra_figures.py` — methodology / forest / per-prompt figures.
- `scripts/make_appendix.py` — transparency appendix (all prompts + responses).
- `scripts/reliability.py` — human-vs-LLM-judge agreement.
- `scripts/make_factsheet.py` — `RESULTS_FACTSHEET.md`.
- `scripts/llm_judge.py` — optional automated second rater (needs API key).
- `tests/` — unit tests for stats, retrieval, data integrity.
- `paper/` — LaTeX skeleton + auto-generated tables/figures/macros + references.

## Reproduce
With `results/model_outputs.csv` present and `results/scoring_sheet.csv` filled:
```
bash run_all.sh
```
(`run_experiment.py` and `llm_judge.py` are the only steps that need an API key.)

## Measures (1-5 human ratings)
- sycophancy — agrees with / amplifies the user's framing (lower better)
- stability — calm, steady reasoning (higher better)
- process — effort / control orientation (higher better)

## Limitations to address in the paper (your words)
- Small dataset (28 verses) and prompt set (14).
- Subjective scoring; single human rater (mitigated by blinding + LLM judge).
- A few RAG responses explicitly name "the Gita," slightly weakening blinding.
- One model; limited statistical power (treat trends as exploratory).
- Hand-crafted lexicons for the objective metrics are exploratory.

## Integrity note
The study design, all human judgments (verse selection, prompt set, blind
ratings), and the paper are the author's. The code, statistics, figures, and
organization were built with AI assistance (like using a stats/plotting library
or a research assistant for tooling). Disclose AI assistance per your venue's
policy.
