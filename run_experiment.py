"""Run the Nishkama-AI experiment.

For each emotionally charged prompt we generate two responses from the SAME
language model:

  * Baseline : prompt -> model (no external context)
  * Nishkama : retrieve 1-2 Bhagavad Gita verses, prepend them to the prompt,
               then prompt -> model

The only difference between the two conditions is the retrieved verse context,
so any behavioural difference is attributable to that context alone.

The runner is resilient: it retries transient API errors (e.g. 529 Overloaded)
with backoff, and it saves progress after every prompt, so a crash never loses
work -- just re-run the same command and it resumes where it left off.

Usage
-----
    export ANTHROPIC_API_KEY="sk-ant-..."
    python run_experiment.py                 # full run, all prompts (resumable)
    python run_experiment.py --limit 3       # quick smoke test (3 prompts)
    python run_experiment.py --dry-run       # no API calls; inspect retrieval
    python run_experiment.py --model claude-sonnet-4-6   # cheaper model
    python run_experiment.py --retrieval tfidf           # force a retriever
    python run_experiment.py --fresh         # ignore any existing output, start over

Output -> results/model_outputs.csv
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time

import numpy as np
import pandas as pd

import config

try:  # optional at import time so --dry-run works without the SDK installed
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


class Retriever:
    """Scores Gita verses against a query. Tries, in order of preference:

    1. semantic  : sentence-transformers embeddings + cosine similarity
    2. tfidf     : scikit-learn TF-IDF + cosine similarity
    3. lexical   : token-overlap (always available, no dependencies)

    The actual method used is recorded so it can be reported in the paper.
    """

    def __init__(self, texts: list[str], method: str = "auto") -> None:
        self.texts = texts
        self.method = None
        order = ["semantic", "tfidf", "lexical"] if method == "auto" else [method]
        for candidate in order:
            try:
                getattr(self, f"_init_{candidate}")()
                self.method = candidate
                break
            except Exception as exc:  # noqa: BLE001 - fall back gracefully
                if method != "auto":
                    raise
                print(f"[retrieval] '{candidate}' unavailable ({exc}); falling back.")
        if self.method is None:
            raise RuntimeError("No retrieval method could be initialised.")
        print(f"[retrieval] using method: {self.method}")

    # -- semantic -----------------------------------------------------------
    def _init_semantic(self) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import

        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = self._model.encode(self.texts, convert_to_numpy=True)
        # float64 + explicit cosine avoids float32 matmul warnings/overflow on
        # some platforms while keeping the ranking identical.
        self._corpus = np.asarray(emb, dtype=np.float64)
        self._corpus_norms = np.linalg.norm(self._corpus, axis=1)

    def _score_semantic(self, query: str) -> np.ndarray:
        q = np.asarray(
            self._model.encode([query], convert_to_numpy=True)[0], dtype=np.float64
        )
        denom = self._corpus_norms * np.linalg.norm(q)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scores = (self._corpus @ q) / denom
        return np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    # -- tfidf --------------------------------------------------------------
    def _init_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(stop_words="english")
        self._corpus = self._vec.fit_transform(self.texts)

    def _score_tfidf(self, query: str) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity

        qv = self._vec.transform([query])
        return cosine_similarity(qv, self._corpus)[0]

    # -- lexical ------------------------------------------------------------
    def _init_lexical(self) -> None:
        self._tok = [set(_tokenize(t)) for t in self.texts]

    def _score_lexical(self, query: str) -> np.ndarray:
        q = set(_tokenize(query))
        scores = []
        for toks in self._tok:
            if not toks or not q:
                scores.append(0.0)
            else:
                scores.append(len(q & toks) / len(q | toks))
        return np.asarray(scores)

    # -- public -------------------------------------------------------------
    def retrieve(self, query: str, k: int) -> list[int]:
        scores = getattr(self, f"_score_{self.method}")(query)
        return [int(i) for i in np.argsort(scores)[::-1][:k]]


# ---------------------------------------------------------------------------
# Generation (with retry on transient errors)
# ---------------------------------------------------------------------------
def build_context_block(rows: pd.DataFrame) -> str:
    lines = []
    for i, row in enumerate(rows.itertuples(index=False), start=1):
        lines.append(f"[{i}] ({row.reference}) {row.text}")
    return "\n".join(lines)


def generate(client, model: str, user: str, max_attempts: int = 6) -> str:
    """Call the model, retrying transient errors (529/overload, rate limit, 5xx,
    connection drops) with exponential backoff."""
    transient = (
        anthropic.OverloadedError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
    )
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=config.DEFAULT_MAX_TOKENS,
                system=config.BASE_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except transient as exc:  # noqa: PERF203
            if attempt == max_attempts:
                raise
            wait = min(2.0 * (2 ** (attempt - 1)) + random.uniform(0, 1), 30.0)
            print(f"    (transient error: {type(exc).__name__}; "
                  f"retry {attempt}/{max_attempts - 1} in {wait:.0f}s)")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Nishkama-AI experiment.")
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N prompts (smoke test).")
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat each prompt this many times.")
    parser.add_argument("--retrieval", default="auto",
                        choices=["auto", "semantic", "tfidf", "lexical"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Do retrieval but skip API calls (no key needed).")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing output file and start over.")
    parser.add_argument("--output", default=str(config.MODEL_OUTPUTS_CSV))
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prompts = pd.read_csv(config.PROMPTS_CSV)
    gita = pd.read_csv(config.GITA_CSV)
    if args.limit:
        prompts = prompts.head(args.limit)

    retriever = Retriever(gita["text"].tolist(), method=args.retrieval)

    client = None
    if not args.dry_run:
        if anthropic is None:
            print("ERROR: `anthropic` not installed. Run `pip install anthropic`, "
                  "or use --dry-run.", file=sys.stderr)
            return 1
        try:
            client = anthropic.Anthropic(max_retries=8)  # SDK also backs off
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR creating Anthropic client: {exc}\n"
                  "Set ANTHROPIC_API_KEY, or use --dry-run.", file=sys.stderr)
            return 1

    # Resume support: load any already-completed (run, id) pairs.
    from pathlib import Path
    records: list[dict] = []
    done: set[tuple[int, str]] = set()
    out_path = Path(args.output)
    if out_path.exists() and not args.fresh:
        prev = pd.read_csv(out_path)
        records = prev.to_dict("records")
        done = {(int(r["run"]), str(r["id"])) for r in records}
        if done:
            print(f"[resume] found {len(done)} completed items in {out_path}; "
                  "skipping those. (Use --fresh to start over.)")

    columns = ["run", "id", "category", "prompt", "model", "retrieval_method",
               "retrieved_ids", "retrieved_refs", "retrieved_text",
               "baseline_response", "nishkama_response"]

    total = len(prompts) * args.runs
    done_count = len(done)
    for run in range(1, args.runs + 1):
        for row in prompts.itertuples(index=False):
            if (run, row.id) in done:
                continue
            idx = retriever.retrieve(row.prompt, args.top_k)
            chosen = gita.iloc[idx]
            context = build_context_block(chosen)
            rag_user = config.RAG_USER_TEMPLATE.format(context=context, prompt=row.prompt)

            if args.dry_run:
                baseline_resp = "[DRY RUN - no API call]"
                nishkama_resp = "[DRY RUN - no API call]"
            else:
                baseline_resp = generate(client, args.model, row.prompt)
                nishkama_resp = generate(client, args.model, rag_user)

            records.append({
                "run": run,
                "id": row.id,
                "category": row.category,
                "prompt": row.prompt,
                "model": args.model,
                "retrieval_method": retriever.method,
                "retrieved_ids": "; ".join(chosen["id"].tolist()),
                "retrieved_refs": "; ".join(chosen["reference"].tolist()),
                "retrieved_text": " | ".join(chosen["text"].tolist()),
                "baseline_response": baseline_resp,
                "nishkama_response": nishkama_resp,
            })
            # Save after every prompt so a crash never loses progress.
            pd.DataFrame(records).to_csv(out_path, index=False, columns=columns)

            done_count += 1
            print(f"[{done_count}/{total}] run {run} {row.id} ({row.category}) "
                  f"-> verses {', '.join(chosen['reference'].tolist())}")

    print(f"\nDone. Wrote {len(records)} rows to {out_path}")
    if args.dry_run:
        print("This was a DRY RUN. Re-run without --dry-run (with an API key) "
              "to generate real responses.")
    else:
        print("Next: build the scoring sheet -> "
              "python scripts/make_scoring_template.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
