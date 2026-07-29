"""
Example 1 — Effort knobs (mode-conditioned reasoning effort).

Sweeps a small benchmark through Nemotron's "detailed thinking off" vs "on" toggle and
records, per problem, the compute spent (reasoning chars, completion tokens, latency) and
whether the final answer is correct. This is the two-sided evidence for the blog: the same
model, same weights, a system-prompt knob that trades tokens for accuracy — reasoning effort
as an engineered control, not raw capability.

Model: nvidia/llama-3.3-nemotron-super-49b-v1 (the variant that honors the toggle cleanly;
see llm_client.DEFAULT_MODEL). Emits data/phase3/effort_modes.csv.

Run:
    python -m src.demos.effort_modes           # full sweep (both efforts, ~10 min)
    python -m src.demos.effort_modes --quick   # first 4 problems only
"""

from __future__ import annotations

import argparse
import csv
import os
import re

from src.demos.llm_client import call, DEFAULT_MODEL

# Handcrafted set: unambiguous integer answers, spanning trivial -> multi-step so the
# effort/accuracy tradeoff is visible (reasoning should help most on the harder ones).
PROBLEMS = [
    {"id": "mult",      "answer": 1081, "prompt": "What is 23 * 47? Answer with just the number."},
    {"id": "sheep",     "answer": 9,    "prompt": "A farmer has 17 sheep. All but 9 run away. How many are left? Answer with just the number."},
    {"id": "percent",   "answer": 36,   "prompt": "What is 15% of 240? Answer with just the number."},
    {"id": "discount",  "answer": 50,   "prompt": "A shirt costs $40 after a 20% discount. What was the original price in dollars? Answer with just the number."},
    {"id": "even_sum",  "answer": 110,  "prompt": "What is the sum of the first 10 positive even numbers? Answer with just the number."},
    {"id": "perimeter", "answer": 40,   "prompt": "A rectangle is 12 by 8. What is its perimeter? Answer with just the number."},
    {"id": "minutes",   "answer": 210,  "prompt": "How many minutes are in 3.5 hours? Answer with just the number."},
    {"id": "algebra",   "answer": 8,    "prompt": "A number tripled and then increased by 6 equals 30. What is the number? Answer with just the number."},
    {"id": "chain",     "answer": 60,   "prompt": "What is 144 divided by 12, then multiplied by 5? Answer with just the number."},
    {"id": "fraction",  "answer": 15,   "prompt": "There are 24 students and 3/8 of them are boys. How many are girls? Answer with just the number."},
]

OUT_DIR = "data/phase3"
OUT_CSV = f"{OUT_DIR}/effort_modes.csv"


def extract_int(text: str):
    """Pull the last integer out of a response (handles trailing prose in 'on' mode)."""
    nums = re.findall(r"-?\d[\d,]*", text or "")
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None


def run_sweep(problems, model: str = DEFAULT_MODEL) -> list[dict]:
    rows = []
    for p in problems:
        for effort in ("off", "on"):
            r = call(p["prompt"], effort=effort, model=model, max_tokens=2048)
            pred = extract_int(r.content)
            correct = pred == p["answer"]
            rows.append({
                "problem_id": p["id"],
                "effort": effort,
                "model": model,
                "expected": p["answer"],
                "predicted": pred,
                "correct": int(correct),
                "reasoning_chars": r.reasoning_chars,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "latency_s": r.latency_s,
            })
            print(f"  {p['id']:10s} [{effort:3s}] pred={pred} correct={correct} "
                  f"tokens={r.completion_tokens} reasoning_chars={r.reasoning_chars}")
    return rows


def summarize(rows):
    for effort in ("off", "on"):
        sub = [r for r in rows if r["effort"] == effort]
        n = len(sub)
        acc = sum(r["correct"] for r in sub) / n if n else 0
        mean_tok = sum(r["completion_tokens"] for r in sub) / n if n else 0
        mean_lat = sum(r["latency_s"] for r in sub) / n if n else 0
        print(f"[{effort:3s}] accuracy={acc:.0%}  mean_completion_tokens={mean_tok:.0f}  "
              f"mean_latency={mean_lat:.1f}s  (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 4 problems only")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    problems = PROBLEMS[:4] if args.quick else PROBLEMS
    print(f"Effort sweep: {len(problems)} problems x 2 efforts on {DEFAULT_MODEL}")
    rows = run_sweep(problems)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")
    summarize(rows)


if __name__ == "__main__":
    main()
