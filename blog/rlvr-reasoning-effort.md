# I trained a 1.5B model to reason — then measured what "thinking" actually costs

Reasoning models feel like magic until you build one. Then you notice something: the "thinking" isn't a personality trait the model was born with. It's a behavior that gets **trained in**, **priced in tokens**, and **turned up or down at inference** like any other setting.

So I spent a weekend making that concrete. I trained a small model to reason from scratch on a single GPU, watched it learn, looked at how the same trick can go wrong, and then measured what it costs to dial its reasoning up and down. Everything below is from that run — one training job on an NVIDIA L4, plus three inference experiments against a hosted reasoning model. The numbers are mine unless a figure says otherwise.

---

## The setup: teach a 1.5B model to solve Sudoku

The cleanest way to watch reasoning get trained is **RLVR** — Reinforcement Learning from Verifiable Rewards. Instead of a learned reward model that can be gamed, the reward is a deterministic check: *did the model produce a valid solution?* For a task with a right answer, that check is exact and impossible to sweet-talk.

My task was 4×4 mini-Sudoku. The policy is **Qwen2.5-1.5B-Instruct** with a LoRA adapter. Each step, the model generates a group of candidate solutions; a NeMo Gym verifier scores each one; **GRPO** turns those scores into group-relative advantages and nudges the adapter toward the answers that scored well. No critic network, no human labels, no reward model to hack.

![The RLVR loop](figures/07_architecture.png)

Two numbers made this cheap enough to do on a whim: I trained **only 2.3% of the model** (36.9M LoRA parameters), and peak memory stayed at **14.7 GB** — comfortably inside a 24 GB L4. Full fine-tuning runs out of memory before the first step; LoRA plus a right-sized sequence length is what makes it fit.

![Why one cheap GPU was enough](figures/02_efficiency.png)

## Watching reasoning get trained in

Over **100 GRPO steps — about 76 minutes and ~$2 of GPU time** — the verifier reward climbs from **0.09 to 0.55**. The model is never shown a single solution. It *discovers*, through trial and reward, how to produce grids that pass the check.

![Training dynamics](figures/01_training_dynamics.png)

The secondary signals are the fun part. KL divergence stays low, so the policy never lurches off a cliff. Completion length *drops* as training goes on — the model stops rambling and finds terse solutions. And the reward spread narrows, which is the model getting **surer of itself**, not just luckier. That top curve is the whole thesis in one line: reasoning ability here isn't innate. It's the output of an optimization you can watch happen, step by step, against a reward you defined.

## The catch: rewards are gameable

I used a deterministic verifier precisely because it's hard to fool. Swap in a reward that's only a *proxy* for correctness — a learned scorer, a format bonus, a length reward — and optimization will find the gap between the proxy and the truth. **Reward goes up; real accuracy goes down.** The model has learned to satisfy the metric instead of doing the task.

![When the reward is gameable, accuracy diverges](figures/06_reward_hacking.png)

This is the failure mode behind a lot of "we trained it and it somehow got worse" stories (the curve above is illustrative, after [Wang et al., arXiv:2505.22203](https://arxiv.org/abs/2505.22203)). The practical takeaway is narrow and useful: **the closer your reward is to verifiable ground truth, the less room the model has to cheat.** Where you can afford a deterministic check, use one — my run doesn't hack, because there's nothing to hack.

## Turning reasoning up and down at inference

Training decides whether a model *can* reason. Inference decides how much it *does* on any given request — and that's where the real control surface lives. Three experiments, all against a hosted Nemotron reasoning model, show three different knobs.

### Knob 1 — an effort switch

The model takes a "detailed thinking on / off" system prompt. Same weights, same questions, one line of config. Across ten problems, **"off" answers in 2 tokens on average; "on" spends 372.** On the easy ones the answers are identical — the reasoning is pure overhead. On the harder ones, "off" slips to 80% while "on" holds 100%.

![Reasoning on demand](figures/03_effort_modes.png)

Every gray dot is a 2-token answer; every colored dot is the same question with reasoning on. The two red dots are the problems where thinking actually *changed the answer* from wrong to right. So the switch isn't "smarter vs dumber" — it's **cheap vs thorough**, and the right setting depends entirely on the problem. Paying 186× the tokens to get the same answer to `23 × 47` isn't intelligence. It's a misconfigured dial.

### Knob 2 — a hard budget

You can also cap thinking mid-stream. I forced the model to stop reasoning after a fixed number of tokens by injecting the end-of-thinking marker, then made it answer immediately (the "budget forcing" idea from [s1, arXiv:2501.19393](https://arxiv.org/abs/2501.19393)). Sweeping the cap reveals a **floor**: below ~400 thinking tokens these problems collapse to 0–33%; at 400 they recover to 83%, and removing the cap entirely does no better.

![Budget forcing](figures/04_budget_forcing.png)

The heatmap says it best. The right side goes green; the left is mostly red; the 200-token column is *all* red (truncating there lands worst of all); and `garden` — the hardest problem — never solves at any budget I tried. There's a **compute floor for correctness and a ceiling past which more thinking is wasted**, and both are things you *set*, not properties of how smart the model is.

### Knob 3 — the model sets its own budget

Left uncapped, a good reasoning model already scales its own effort to difficulty. Running a graded ladder from one-step to five-step problems, the reasoning it spends climbs from ~200 tokens to nearly 1000.

![Adaptive compute](figures/05_adaptive_compute.png)

It's not a perfect controller — trivial arithmetic sometimes triggers a long trace, because the model doesn't *know* a problem is easy until it has thought about it. But the shape is right, and it points at where inference-efficiency work actually pays off: not making the model think less everywhere, but stopping it from **over-thinking the easy cases**.

## What this means when you ship one

Put the four measurements together and "reasoning effort" stops looking like a fixed trait and starts looking like what it is — **a control plane you configure**:

- **Whether a model can reason at all is trained in** — and you can watch it happen, against a reward you define.
- **That reward has to be close to ground truth**, or optimization games it.
- **At inference, effort is a switch, a budget, and an automatic response to difficulty** — three levers, all yours to set.

If you're putting reasoning models into production, the leverage is in that second half. The model's raw capability is mostly fixed once it ships. The token bill, the latency, and a good chunk of the accuracy are **not** — they're the dial. Treat them like one.

---

### What I actually ran

- **Training:** Qwen2.5-1.5B-Instruct, LoRA r=32, GRPO via Unsloth + TRL, NeMo Gym deterministic Sudoku verifier, one NVIDIA L4 on Modal. 100 steps, ~76 min, ~$2. Reward 0.09 → 0.55. Every metric above was logged live to **Weights & Biases**; the trained adapter lives on **Hugging Face**.
- **Inference experiments:** `nvidia/llama-3.3-nemotron-super-49b-v1` via the NVIDIA Build API; budget forcing on `DeepSeek-R1-Distill-Qwen-1.5B` (a model whose decode loop I could control).
- **Dataset:** 4×4 mini-Sudoku, 2,000 examples.
- **Honesty note:** the reward-hacking figure is illustrative (after arXiv:2505.22203) — my deterministic-verifier run does *not* exhibit hacking, which is the point.

*Sources: GRPO — DeepSeekMath ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)); budget forcing — s1 ([arXiv:2501.19393](https://arxiv.org/abs/2501.19393)); reward hacking ([arXiv:2505.22203](https://arxiv.org/abs/2505.22203)).*
