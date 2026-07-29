"""
Example 3 — Adaptive compute (reasoning scales with difficulty).

Runs a graded ladder of problems (1 -> 5 required solution steps) through Nemotron with
reasoning ON, and measures how much thinking the model spends at each difficulty. The point
for the blog: a capable reasoning model *allocates* compute to difficulty on its own — trivial
problems get a short trace, multi-step ones get a long one. This feeds Artifact 4
(scatter: X = difficulty/steps, Y = reasoning tokens).

Difficulty is the number of sequential operations the problem requires — an objective,
correct-by-construction X-axis (no Sudoku solver needed; answers are hand-verified).
Model: nvidia/llama-3.3-nemotron-super-49b-v1. Emits data/phase3/adaptive_compute.csv.

Run:
    python -m src.demos.adaptive_compute
"""

from __future__ import annotations

import csv
import os
import re

from src.demos.llm_client import call, DEFAULT_MODEL

# Ladder: 2 problems per difficulty level (steps=1..5). Answers hand-verified.
LADDER = [
    {"id": "L1a", "steps": 1, "answer": 15,  "prompt": "What is 8 + 7? Answer with just the number."},
    {"id": "L1b", "steps": 1, "answer": 54,  "prompt": "What is 9 * 6? Answer with just the number."},
    {"id": "L2a", "steps": 2, "answer": 60,  "prompt": "What is (12 + 8) * 3? Answer with just the number."},
    {"id": "L2b", "steps": 2, "answer": 50,  "prompt": "What is 100 - 35 - 15? Answer with just the number."},
    {"id": "L3a", "steps": 3, "answer": 45,  "prompt": "A store has 120 apples. It sells 45 in the morning and 30 in the afternoon. How many remain? Answer with just the number."},
    {"id": "L3b", "steps": 3, "answer": 65,  "prompt": "What is 15 * 4 + 20 / 4? Answer with just the number."},
    {"id": "L4a", "steps": 4, "answer": 70,  "prompt": "A train travels 60 km in the first hour, 80 km in the second, and 70 km in the third. What is its average speed in km/h over the three hours? Answer with just the number."},
    {"id": "L4b", "steps": 4, "answer": 4,   "prompt": "Sarah buys 3 notebooks at $4 each and 2 pens at $2 each. She pays with a $20 bill. How much change in dollars does she get? Answer with just the number."},
    {"id": "L5a", "steps": 5, "answer": 116, "prompt": "A rectangular garden is 15 m by 10 m. A path 2 m wide surrounds it on all sides. What is the area of the path in square meters? Answer with just the number."},
    {"id": "L5b", "steps": 5, "answer": 330, "prompt": "A tank holds 500 liters and is currently 60% full. Then 80 liters are added and 50 liters are drained. How many liters are in the tank now? Answer with just the number."},
]

OUT_DIR = "data/phase3"
OUT_CSV = f"{OUT_DIR}/adaptive_compute.csv"


def extract_int(text: str):
    nums = re.findall(r"-?\d[\d,]*", text or "")
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None


def run_ladder(model: str = DEFAULT_MODEL) -> list[dict]:
    rows = []
    for p in LADDER:
        r = call(p["prompt"], effort="on", model=model, max_tokens=3072)
        pred = extract_int(r.content)
        correct = pred == p["answer"]
        rows.append({
            "problem_id": p["id"],
            "steps": p["steps"],
            "model": model,
            "expected": p["answer"],
            "predicted": pred,
            "correct": int(correct),
            "reasoning_chars": r.reasoning_chars,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.total_tokens,
            "latency_s": r.latency_s,
        })
        print(f"  {p['id']} steps={p['steps']} pred={pred} correct={correct} "
              f"reasoning_chars={r.reasoning_chars} completion_tokens={r.completion_tokens} "
              f"latency={r.latency_s}s")
    return rows


def summarize(rows):
    print("\nsteps -> mean reasoning tokens (adaptive-compute signal):")
    for s in sorted(set(r["steps"] for r in rows)):
        sub = [r for r in rows if r["steps"] == s]
        mean_tok = sum(r["completion_tokens"] for r in sub) / len(sub)
        mean_chars = sum(r["reasoning_chars"] for r in sub) / len(sub)
        acc = sum(r["correct"] for r in sub) / len(sub)
        print(f"  steps={s}: mean_completion_tokens={mean_tok:.0f}  "
              f"mean_reasoning_chars={mean_chars:.0f}  accuracy={acc:.0%}")


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print(f"Adaptive-compute ladder: {len(LADDER)} problems on {DEFAULT_MODEL}")
    rows = run_ladder()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")
    summarize(rows)


if __name__ == "__main__":
    main()
