# Execution Plan — RLVR Reasoning-Effort Pipeline (Blog Project)

**Goal:** Demonstrate that *reasoning effort is an engineered inference control plane, not raw
intelligence* (Raschka's thesis), using the NVIDIA/Unsloth/Nemotron stack — training a lightweight LLM
(Qwen2.5-1.5B-Instruct) on Sudoku via GRPO and showing how models are trained to think, how they hack
rewards, and how their compute budget is controlled. Output: a Blend blog post + 4 data artifacts.

> Status: **Scoping complete, grounded against primary sources (Jul 2026).** This is the engineering
> plan. The blog visual/written artifacts are a separate, brand-governed deliverable (see Brand checkpoint).

> **Approach: "Hybrid — pay once."** Keep the full NVIDIA/Unsloth/Nemotron stack, but use each tool
> where it is free, and rent a GPU exactly **once** for the single run that produces original evidence.
> Everything else is free-tier API or published data. **Out-of-pocket target: ~$0** (covered by $30
> Modal credits). The original 4-run full-training plan is preserved in the Appendix.

---

## 0. TL;DR

- The core loop already exists: **`NeMo-Gym-Sudoku.ipynb`** (Qwen2.5-1.5B, 4×4 Sudoku, GRPO + Unsloth +
  NeMo Gym) — https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/NeMo-Gym-Sudoku.ipynb
- We run it **once** on **Modal** (free $30 credits) to get a first-hand VRAM log + reward curve + adapter.
- **NeMo Data Designer** generates the dataset (free). **Nemotron** (free inference API) is the live model
  for the effort/budget/adaptive demos. No re-training of things that are already trained or published.

---

## 1. The hybrid — where each stack piece is used, and what it costs

| Stack piece | Role in the blog | Cost |
|---|---|---|
| **NeMo Data Designer** | Schema-driven generation of the Sudoku/logic-puzzle dataset — a real, showcased use of the NVIDIA data stack | **Free** (build.nvidia.com key or local SDK), no GPU |
| **Unsloth + NeMo Gym** | **One real GRPO run** — Unsloth memory efficiency + NeMo Gym Sudoku verifier → your own VRAM log + reward curve + LoRA adapter | **~$10–16 of Modal credit** (one run) |
| **Nemotron** (reasoning) | The live model for Examples 1–3 (effort modes / budget forcing / adaptive compute) | **Free** inference credits (build.nvidia.com) |
| **Published papers** | Reward-hacking divergence curve (Artifact 2); GRPO/budget-forcing/adaptive-compute grounding | **Free** (cite + recreate) |

All three brands appear **authentically**; GPU is rented once.

---

## 2. Compute — Modal (free credits)

- **You have $30 Modal Starter credits.** Modal is serverless: we wrap the training as a Modal function
  (`@app.function(gpu=...)`), checkpoint to a `modal.Volume` + push to HF Hub, run detached (`modal run --detach`).
- **Modal GPU rates (verify live):** T4 ~$0.59/hr · **L4 (24GB, Ada, bf16) ~$0.80/hr** · A100-40GB ~$2.10/hr · H100 ~$3.95/hr.
- **Recommended GPU:** **L4 24GB** — bf16 support (no T4 fp16 compromise), 24GB holds 1.5B training + the
  vLLM 16-bit rollout copy with headroom. A ~12–18h run ≈ **$10–15**. For a faster run, **A100-40GB** (~6–9h ≈ $13–19).
- **Budget headroom:** $30 comfortably covers the baseline run **plus** an optional second run (the
  low/high-effort adapter for Example 1) — see §5 Example 1. Two L4 runs ≈ $20–30.
- **Fallback if Modal's serverless wrapping is fiddly:** a Vast.ai/Salad RTX 4090 (~$0.16–0.40/hr, ~$3–6/run)
  via plain SSH. Keep as plan B; prefer Modal since it's free.

---

## 3. API keys needed to run this  ← (the checklist you asked for)

| # | Key / account | Needed for | Cost | Required? |
|---|---|---|---|---|
| 1 | **Modal** account + API token (`modal token new`) | The single GPU training run | Free ($30 credits) | ✅ Required |
| 2 | **Hugging Face** access token (`HF_TOKEN`) | Download Qwen2.5-1.5B-Instruct; push checkpoints/adapter (Modal storage is ephemeral) | Free | ✅ Required |
| 3 | **NVIDIA Build** API key (build.nvidia.com) | **Both** NeMo Data Designer (dataset gen) **and** Nemotron inference (live demos) — one key, two uses | Free | ✅ Required |
| 4 | **Weights & Biases** key (personal) | Log reward + ground-truth accuracy during the run | Free personal tier | 🟡 Recommended (⚠️ non-corporate-use — confirm for Blend) |
| 5 | **OpenRouter** (or OpenAI) key | Only if we want a *second, non-Nemotron* model for cross-comparison, or a separate LLM-judge | Usage-based | ⚪ Optional (skippable — Nemotron covers the demos) |

**Net: 3 required (Modal, Hugging Face, NVIDIA Build) + WandB recommended.** OpenRouter is no longer needed
unless we add a cross-model comparison — the earlier LLM-judge requirement is dropped because the
reward-hacking artifact now uses the published curve, not a live judged run.

---

## 3a. Pre-flight verification — DONE (2026-07-27)

Live-checked before any GPU spend, so the paid run doesn't fail on a bad assumption:

| Check | Result |
|---|---|
| NVIDIA Build key works | ✅ 102 models, incl. `nvidia/llama-3.3-nemotron-super-49b-v1.5` + `nemotron-3-nano-...-reasoning` |
| Nemotron "detailed thinking on/off" toggle (Example 1 premise) | ✅ Confirmed live — `on` → 3,898-char reasoning trace; answer + reasoning are **separate fields** (`content` vs `reasoning_content`) |
| Hugging Face token valid + **write** scope | ✅ Read + write both confirmed (repo create/delete = 200) → checkpoint push is safe |
| Modal auth + secret | ✅ `~/.modal.toml` profile `samkoduru` active; Modal secret `llm-api-keys` created |
| AWS Secrets Manager | ✅ All 5 keys stored + round-trip verified in `sandbox/llm-api-keys` |
| WandB key | ✅ Stored (Modal + AWS SM); wiring in §WandB below |

**Refinement surfaced by the toggle test:** the hosted Nemotron API returns reasoning in a separate
`reasoning_content` field and does **not** let us inject a `</think>` token mid-generation server-side.
So **Example 2 (budget forcing)** must run against a model whose **decode loop we own** — the vLLM-served
trained adapter (cloud) or a local reasoning GGUF via llama.cpp — not the hosted API. Examples 1 & 3 only
*observe/measure*, so the hosted Nemotron API is fine for those. (Plan updated in §5.)

## WandB — scoped & wired

- **Where:** Phase 2 GRPO run logs to WandB (`report_to="wandb"` in the trainer + explicit
  `wandb.log()` per step). Key flows in via the Modal secret automatically.
- **What we log:** each reward component, mean completion / `<think>` length, and — logged side-by-side —
  **ground-truth accuracy from a held-out deterministic verifier**. That paired reward-vs-accuracy series is
  the reusable asset (proves reward-climb for Example 1; is the template for the optional reward-hacking run).
- **⚠️ Licensing decision (open):** the personal key is a **free tier = non-corporate-use**. For a Blend-branded
  post either (a) use a Blend WandB team seat, or (b) treat this as personal learning with private logs. Your call —
  flag before we publish anything sourced from it.

## 4. Phased execution

**Phase 1 — Dataset (free, ~half day).**
NeMo Data Designer (NVIDIA Build key): schema-driven generation of a puzzle set (4×4 + graded 9×9 for the
difficulty ladder). Validate the async pipeline. *(The Sudoku env + reasoning-gym can also supply puzzles;
Data Designer is the showcased NVIDIA-stack path.)*

**Phase 2 — The one training run (Modal, ~$10–16 credit, unattended overnight).**
Port `NeMo-Gym-Sudoku.ipynb` into a Modal function on **L4 24GB**. Set `HF_TOKEN`, `WANDB_API_KEY`,
`UNSLOTH_VLLM_STANDBY=1`. Baseline config: Qwen2.5-1.5B-Instruct, LoRA r=32, `max_seq_length≈1024`
(prompt 256 / gen 768), `num_generations=8`, `fast_inference=True`, `gpu_memory_utilization≈0.9`.
Capture: **VRAM log** (Artifact 1), **reward curve** (Example 1 authenticity), **LoRA adapter** (→ GGUF for
local Mac inference). Pin versions (`unsloth==2026.1.4` etc.) — the Unsloth↔NeMo-Gym integration is version-sensitive.

> **Money guardrails (mandatory — this is the only paid step):**
> 1. **Smoke-test gate first.** Run `max_steps=5` on the same L4 (~10–15 min, <$0.25). Only if it
>    completes — NeMo Gym servers start, reward is non-zero, a checkpoint pushes to HF — do we launch the
>    full run. This catches version/OOM/env breakage for pennies instead of hours. **The single biggest saver.**
> 2. **Checkpoint to HF every ~25 steps.** Modal storage is ephemeral; a preemption then costs minutes, not the run.
> 3. **Hard cap:** set the Modal function `timeout` (e.g. 18h) and a `max_steps` ceiling so nothing runs away and drains all $30.
> 4. **Cost math:** L4 @ ~$0.80/hr → smoke test ~$0.25, full run ~$10–15. Leaves ≥$14 of the $30 for the optional Example-1 second run.

**Phase 3 — Live demos vs Nemotron (free, ~1 day).**
One shared `llm_client.py` (NVIDIA Build endpoint) with token-usage capture. Three scripts (see §5) →
each emits a CSV feeding an artifact.

**Phase 4 — Artifacts + write (~2 days).**
Build the 4 artifacts (Artifact 1 now = *your* data), diagram, prose. **Engage Blend brand system here.**

**Total: ~$0 out-of-pocket (within $30 Modal), GPU rented once, ~4 days.**

---

## 5. The three examples (hybrid implementation)

**Example 1 — Effort knobs (mode-conditioned effort).** Two-sided evidence:
- *Live:* **Nemotron reasoning toggle** — "detailed thinking on" vs "off" system prompt — the effort knob,
  already trained. Measure token spend + accuracy on the same prompts. ⚠️ *Verify the exact toggle syntax
  on the current Nemotron model card before wiring.*
- *First-hand (optional 2nd Modal run, ~$10):* train a **low-effort LoRA** (penalize long `<think>`) to
  show the mechanism yourself. Keep correctness reward dominant (naive length penalties are hackable /
  cause collapse — AALC, arXiv:2506.20160). Do this only if the post needs the "we trained the knob" moment.

**Example 2 — Budget forcing.** Inject `</think>` after N tokens (e.g. 200) to force an answer on
incomplete reasoning (s1 "budget forcing", arXiv:2501.19393). ⚠️ **Requires a model whose decode loop we
own** — the vLLM-served trained adapter (cloud) or a local reasoning GGUF via llama.cpp — because the hosted
Nemotron API separates `reasoning_content` and won't accept a mid-stream token injection. Capture answer
quality vs. budget. (Verified 2026-07-27 — see §3a.)

**Example 3 — Adaptive compute.** Run Nemotron across the difficulty ladder (trivial 4×4 → sparse 9×9),
log latency + reasoning tokens. Feeds Artifact 4. *(Using an already-capable model sidesteps the
OOD-generalization risk a 4×4-only-trained model would have.)*

---

## 6. The four artifacts (honest capture)

1. **VRAM Reality Check (bar).** From your Phase-2 Modal run: `torch.cuda.max_memory_allocated()` for
   Unsloth; contextualize with Unsloth's published 8B@20K figure (510.8→54.3GB). Now *your* measurement.
2. **Reward Hacking Post-Mortem (line).** Recreate the divergence curve from **arXiv:2505.22203** (reward
   rises, oracle accuracy falls ~450 iters). Cited, not re-trained. *(Optional live version = extra run + judge.)*
3. **Architecture Flow (diagram).** vLLM rollouts → NeMo Gym deterministic verifier → Unsloth z-score
   (group-standardized) advantages → LoRA update. *(z-score std is debated — Dr.GRPO, arXiv:2503.20783 — add a caveat.)*
4. **Compute Allocation (scatter).** X = difficulty (empty cells / entropy), Y = reasoning tokens, from
   Example 3's CSV.

---

## 7. Corrections carried forward (don't regress on these)

- **"Unsloth Studio via MLX" is not a training path** — Unsloth GRPO is CUDA/Triton-only; Mac = inference on
  exported GGUF (LM Studio / Ollama / llama.cpp / mlx-lm).
- **NVIDIA free tiers are inference/lab-only** (build.nvidia.com/NIM, LaunchPad, DLI) — they cannot run custom
  training. That's why the one run goes on Modal, not "free NVIDIA GPU."
- **Reward-hacking gibberish** is a model-based/format/length-reward phenomenon, not a strict deterministic
  verifier one — Artifact 2 uses the published model-based-verifier curve.
- **Don't quote Raschka's "control plane" line verbatim** — it's our paraphrase; lift exact wording from the article.
- **T4 is fp16-only** — moot now (using L4/A100 bf16 on Modal).

## 8. Top risks
1. Modal serverless wrapping of a notebook-style training loop → budget an hour to convert to a script + Volume checkpointing; Vast SSH is plan B.
2. Version drift (Unsloth↔NeMo-Gym) → pin + snapshot.
3. Nemotron toggle syntax → verify on model card before Example 1.
4. WandB corporate-use restriction → resolve before logging company work.

## Brand checkpoint
Blog-facing charts/diagram/prose are a Blend deliverable — engage Blend's brand system and confirm
audience/format before producing them. This plan is internal scoping, intentionally un-styled.

## Primary sources
- Raschka, "Controlling Reasoning Effort in LLMs" — https://magazine.sebastianraschka.com/p/controlling-reasoning-effort-in-llms
- Unsloth GRPO/VRAM — https://unsloth.ai/blog/grpo · https://unsloth.ai/blog/r1-reasoning
- NeMo Gym — https://github.com/NVIDIA-NeMo/Gym · Data Designer — https://github.com/NVIDIA-NeMo/DataDesigner · build.nvidia.com/nemo/data-designer
- Sudoku notebook — https://github.com/unslothai/notebooks/blob/main/nb/NeMo-Gym-Sudoku.ipynb
- Modal pricing — https://modal.com/pricing
- GRPO — DeepSeekMath arXiv:2402.03300 · DeepSeek-R1 arXiv:2501.12948 · Dr.GRPO arXiv:2503.20783
- Budget forcing — s1 arXiv:2501.19393 · Reward hacking — arXiv:2505.22203 · Adaptive — arXiv:2507.02076, arXiv:2510.19669

---

## Appendix — Original full-training plan (4 runs, for reference)

The original scope trained everything firsthand: baseline + low-effort adapter + high-effort adapter +
a deliberate reward-hacking run (~43–65 GPU-hrs, ~$20–80 on RunPod/Vast, ~1 week). Superseded by the
hybrid because most of it re-derives already-trained (Nemotron toggle) or already-published (reward-hacking
curve, VRAM figures) results. Revisit only if the blog pivots to an original-research framing.
