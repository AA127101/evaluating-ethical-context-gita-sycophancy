#!/usr/bin/env bash
# Reproduce the full no-API analysis pipeline in one command.
#
# Assumes you have already (a) run the experiment and (b) filled in the scoring
# sheet -- those two steps need your API key / your human judgments:
#     python run_experiment.py                 # needs ANTHROPIC_API_KEY
#     python scripts/make_scoring_template.py  # then score results/scoring_sheet.csv
#
# Everything below is deterministic and needs no API key.
set -euo pipefail
cd "$(dirname "$0")"

# Use python if present (e.g. inside the venv), else fall back to python3.
PY="$(command -v python || command -v python3)"
echo "Using interpreter: $PY"

echo "==> Running unit tests"
"$PY" -m unittest discover -s tests

echo "==> Statistics, charts, LaTeX tables/macros"
"$PY" analyze_results.py

echo "==> Objective lexical indicators"
"$PY" scripts/objective_metrics.py

echo "==> Extra figures (methodology / forest / per-prompt)"
"$PY" scripts/extra_figures.py

echo "==> Transparency appendix"
"$PY" scripts/make_appendix.py

echo "==> Results fact sheet"
"$PY" scripts/make_factsheet.py

echo ""
echo "Done. Optional next steps that need your API key:"
echo "  python scripts/llm_judge.py        # automated second rater"
echo "  python scripts/reliability.py      # human-vs-judge agreement"
echo "Then write the paper in paper/main.tex and build it on Overleaf."
