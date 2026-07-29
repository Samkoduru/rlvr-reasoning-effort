"""
Example 2 — Budget forcing (s1, arXiv:2501.19393).

Caps a reasoning model's thinking to a token budget by injecting `</think>` mid-stream, then
forces an immediate answer. Sweeping the budget shows the compute/accuracy tradeoff: too small a
budget -> the model answers on incomplete reasoning and gets it wrong; enough budget -> correct.
This is the "reasoning effort is a knob you can turn *down*" side of the story, and unlike the
hosted Nemotron API (separate `reasoning_content`, no mid-stream injection) it requires owning the
decode loop — so we run a local reasoning model whose generation we control.

Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B — same Qwen2.5-1.5B family as our trained Sudoku
model, but a reasoning distill that emits <think>...</think> traces we can truncate. Runs on Modal
(L4) for speed; pure inference, no training deps.

Emits data/phase3/budget_forcing.csv (pulled back from the Modal volume by the local entrypoint).

Usage:
    modal run src/demos/budget_forcing.py
"""

import modal

APP_NAME = "rlvr-budget-forcing"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

app = modal.App(APP_NAME)
outputs = modal.Volume.from_name("rlvr-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate")
)

# Harder multi-step problems (reasoning genuinely matters -> truncating it should hurt accuracy).
# Integer answers, hand-verified.
PROBLEMS = [
    {"id": "garden", "answer": 116, "prompt": "A rectangular garden is 15 m by 10 m. A path 2 m wide surrounds it on all sides. What is the area of the path in square meters?"},
    {"id": "tank",   "answer": 330, "prompt": "A tank holds 500 liters and is currently 60% full. Then 80 liters are added and 50 liters are drained. How many liters are in the tank now?"},
    {"id": "train",  "answer": 70,  "prompt": "A train travels 60 km in the first hour, 80 km in the second, and 70 km in the third. What is its average speed in km/h over the three hours?"},
    {"id": "profit", "answer": 130, "prompt": "A store buys 40 items at $3 each, sells 30 of them at $7 each and the remaining 10 at $4 each. What is the total profit in dollars?"},
    {"id": "fuel",   "answer": 20,  "prompt": "A car uses 8 liters of fuel per 100 km. How many liters does it use on a 250 km trip?"},
    {"id": "change", "answer": 4,   "prompt": "Sarah buys 3 notebooks at $4 each and 2 pens at $2 each. She pays with a $20 bill. How much change in dollars does she get?"},
]

# None = no cap (let the model think to completion, natural baseline).
BUDGETS = [40, 100, 200, 400, None]

OUT_CSV = "/outputs/metrics/budget_forcing.csv"


@app.function(image=image, gpu="L4", volumes={"/outputs": outputs}, timeout=60 * 60)
def run():
    import csv
    import os
    import re
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    think_close_id = tok.encode("</think>", add_special_tokens=False)

    def extract_int(text):
        nums = re.findall(r"-?\d[\d,]*", text or "")
        if not nums:
            return None
        try:
            return int(nums[-1].replace(",", ""))
        except ValueError:
            return None

    def generate(prompt, budget):
        """Budget forcing: cap thinking at `budget` tokens, inject </think>, force the answer.
        budget=None -> think to completion (baseline)."""
        msgs = [{"role": "user", "content": prompt + " Give the final answer as just a number."}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        input_ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        prompt_len = input_ids.shape[1]

        if budget is None:
            out = model.generate(input_ids, max_new_tokens=2048, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = out[0][prompt_len:]
            full = tok.decode(gen, skip_special_tokens=True)
            think_tokens = len(gen)
            forced = False
        else:
            # Phase 1: think up to `budget` tokens.
            out = model.generate(input_ids, max_new_tokens=budget, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = out[0][prompt_len:]
            think_tokens = len(gen)
            decoded = tok.decode(gen, skip_special_tokens=True)
            if "</think>" in decoded:
                # Model finished thinking within budget on its own.
                full = decoded
                forced = False
            else:
                # Still thinking -> force-close and make it answer now.
                forced = True
                forced_ids = torch.cat([
                    out[0].unsqueeze(0),
                    torch.tensor([think_close_id], device="cuda"),
                ], dim=1)
                out2 = model.generate(forced_ids, max_new_tokens=64, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
                full = tok.decode(out2[0][prompt_len:], skip_special_tokens=True)

        # Answer is whatever follows the final </think>.
        answer_text = full.split("</think>")[-1] if "</think>" in full else full
        return extract_int(answer_text), think_tokens, forced

    rows = []
    for p in PROBLEMS:
        for b in BUDGETS:
            t0 = time.time()
            pred, think_tokens, forced = generate(p["prompt"], b)
            correct = pred == p["answer"]
            rows.append({
                "problem_id": p["id"],
                "budget": b if b is not None else "none",
                "expected": p["answer"],
                "predicted": pred,
                "correct": int(correct),
                "thinking_tokens_used": think_tokens,
                "forced": int(forced),
                "latency_s": round(time.time() - t0, 2),
            })
            print(f"  {p['id']:7s} budget={str(b):5s} pred={pred} correct={correct} "
                  f"think_tok={think_tokens} forced={forced}")

    os.makedirs("/outputs/metrics", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    outputs.commit()

    # Accuracy by budget (the headline signal).
    print("\nbudget -> accuracy:")
    for b in BUDGETS:
        key = b if b is not None else "none"
        sub = [r for r in rows if r["budget"] == key]
        acc = sum(r["correct"] for r in sub) / len(sub)
        print(f"  budget={str(key):5s}: accuracy={acc:.0%}")
    return rows


@app.local_entrypoint()
def main():
    rows = run.remote()
    print(f"\nDone: {len(rows)} rows written to volume at {OUT_CSV}")
