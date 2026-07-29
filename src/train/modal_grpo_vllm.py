"""
RLVR Sudoku GRPO on Modal — vLLM fast_inference variant.

Same training recipe as modal_grpo.py (Qwen2.5-1.5B, 4x4 mini-Sudoku, GRPO, LoRA r=32,
seq 768, grad_accum 16, num_generations 8) — the ONLY difference is the generation
backend: this version uses Unsloth `fast_inference=True` (in-process/colocated vLLM)
instead of the HF `.generate()` path. Kept as a separate app so it can't disturb the
proven baseline run, and so we get a clean HF-vs-vLLM comparison (throughput + VRAM)
for the blog.

Key deltas vs the HF path (all grounded in Unsloth's Memory-Efficient-RL docs):
  - image installs vLLM NIGHTLY (wheels.vllm.ai/nightly) because every *released* vLLM
    (<=0.26.0) hard-pins torch==2.11.0 and would downgrade our torch 2.13.0+cu130. The
    nightly (->0.27.0.dev) is the only build linked against torch 2.13.0. torch==2.13.0
    is pinned in the same resolve as a guardrail, and a post-install assert fails the
    build if torch/xformers got clobbered.
  - from_pretrained: fast_inference=True, max_lora_rank=32, gpu_memory_utilization=0.85;
    offload_embedding DROPPED (untested with the vLLM standby path; a known OOM edge).
  - GRPOConfig: use_vllm=True, vllm_mode="colocate".
  - env: UNSLOTH_VLLM_STANDBY=1 (weights shared between trainer + vLLM; only the KV cache
    is time-shared with the optimizer step) and VLLM_WORKER_MULTIPROC_METHOD=spawn (Modal
    + CUDA + fork is a classic hang source).

Usage:
    modal run src/train/modal_grpo_vllm.py --smoke     # 5-step vLLM gate
    modal run --detach src/train/modal_grpo_vllm.py     # full 100-step vLLM run
"""

import modal

APP_NAME = "rlvr-sudoku-grpo-vllm"
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
GYM_DIR = "/root/Gym"
HF_REPO = "Samanthkoduru/rlvr-sudoku-grpo-vllm"        # full-run adapter target
HF_SMOKE_REPO = "Samanthkoduru/rlvr-sudoku-vllm-smoke"  # smoke-run push probe

app = modal.App(APP_NAME)
outputs = modal.Volume.from_name("rlvr-outputs", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: same Unsloth + NeMo Gym base as the HF path, PLUS vLLM nightly on top.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install("uv")
    .pip_install(
        "unsloth",
        "unsloth_zoo",
        "omegaconf",
        "trl",
        "datasets",
        "requests",
        "pyyaml",
        "wandb",
        "huggingface_hub",
    )
    .run_commands(
        f"git clone https://github.com/NVIDIA-NeMo/Gym.git {GYM_DIR}",
        f"cd {GYM_DIR} && uv venv --python 3.12",
        f"cd {GYM_DIR} && bash -c 'source .venv/bin/activate && uv sync'",
        f"cd {GYM_DIR} && bash -c 'source .venv/bin/activate && uv pip install reasoning-gym matplotlib'",
    )
    .run_commands(
        f"cd {GYM_DIR} && bash -c 'source .venv/bin/activate && python "
        "resources_servers/reasoning_gym/scripts/create_dataset.py "
        "--task mini_sudoku --size 2000 --seed 42 "
        "--output resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl'"
    )
    # xformers cu130 fix (same as the HF path — the PyPI wheel is a cu128 build)
    .run_commands(
        "pip install --force-reinstall --no-deps "
        "--index-url https://download.pytorch.org/whl/cu130 xformers==0.0.35",
    )
    # vLLM nightly (the only build linked against torch 2.13; released vLLM pins 2.11).
    # torch 2.13.0+cu130 is ALREADY installed by the unsloth layer, so we only need the
    # install to NOT move it: the explicit `torch==2.13.0` on the command line is the pin
    # that forbids a downgrade. Plain pip avoids uv's --torch-backend (which has no cu130
    # value yet). Then assert the stack survived before we ever spend GPU time.
    .run_commands(
        # vLLM nightly builds are PRE-RELEASE (0.27.0.devN). pip skips pre-releases without
        # --pre and backtracks to ancient vllm 0.2.5; uv with --prerelease=allow +
        # --index-strategy unsafe-best-match actually considers the nightly wheels and picks
        # 0.27.0.dev (which pins torch==2.13.0, so our torch is untouched). torch==2.13.0 is
        # pinned in-line as the guardrail against any downgrade.
        # Pin torch AND transformers: vLLM nightly otherwise bumps transformers to 5.x, which
        # breaks Unsloth 2025.11.1's patching (it pins transformers<=4.57.2 -> auto_docstring
        # NameError). If vLLM nightly hard-requires transformers 5.x this resolve fails fast.
        "uv pip install --system "
        "--prerelease=allow --index-strategy unsafe-best-match "
        "--extra-index-url https://wheels.vllm.ai/nightly "
        "vllm torch==2.13.0 transformers==4.57.2",
        "python -c \"import torch, xformers, transformers; "
        "assert torch.__version__.startswith('2.13.0'), torch.__version__; "
        "assert transformers.__version__ == '4.57.2', transformers.__version__; "
        "import vllm; print('STACK OK', torch.__version__, transformers.__version__, vllm.__version__)\"",
    )
    .env({
        "UNSLOTH_VLLM_STANDBY": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
)


def _start_nemo_gym_server():
    """Start the NeMo Gym resources server and return the /verify endpoint URL."""
    import subprocess
    import time
    import requests
    import yaml
    from omegaconf import OmegaConf

    head_port = 11000
    log = open(f"{GYM_DIR}/ng_run.log", "w")
    subprocess.Popen(
        [
            "bash", "-c",
            "source .venv/bin/activate && ng_run "
            '"+config_paths=[resources_servers/reasoning_gym/configs/resources_only.yaml]" '
            "+uv_pip_set_python=true",
        ],
        cwd=GYM_DIR, stdout=log, stderr=subprocess.STDOUT,
    )

    print("Waiting for NeMo Gym head server", end="", flush=True)
    for _ in range(120):
        try:
            requests.get(f"http://127.0.0.1:{head_port}/global_config_dict_yaml", timeout=2)
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(".", end="", flush=True)
            time.sleep(3)
    else:
        raise RuntimeError(f"Head server did not start; see {GYM_DIR}/ng_run.log")
    print(" ready!")

    resp = requests.get(f"http://127.0.0.1:{head_port}/global_config_dict_yaml", timeout=5)
    resp.raise_for_status()
    cfg = OmegaConf.create(yaml.safe_load(resp.text))
    rs = cfg["reasoning_gym"].resources_servers["reasoning_gym"]

    print(f"Waiting for resources server at {rs.host}:{rs.port}", end="", flush=True)
    for _ in range(90):
        try:
            requests.get(f"http://{rs.host}:{rs.port}/", timeout=2)
            break
        except requests.exceptions.ConnectionError:
            print(".", end="", flush=True)
            time.sleep(2)
    else:
        raise RuntimeError("Resources server did not start within 3 minutes.")
    print(" ready!")
    return f"http://{rs.host}:{rs.port}/verify"


@app.function(
    image=image,
    gpu="L4",
    volumes={"/outputs": outputs},
    secrets=[modal.Secret.from_name("llm-api-keys")],
    timeout=18 * 60 * 60,
    retries=0,
)
def train(smoke: bool = False):
    import os
    # UNSLOTH_VLLM_STANDBY must be set before importing unsloth (also baked into image env).
    os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

    import json
    import numpy as np
    import requests
    import torch

    max_steps = 5 if smoke else 100
    run_name = "vllm-smoke-5step" if smoke else "vllm-baseline-100step"
    hub_repo = HF_SMOKE_REPO if smoke else HF_REPO
    print(f"=== RLVR Sudoku GRPO (vLLM) | run={run_name} | max_steps={max_steps} ===")

    verify_endpoint = _start_nemo_gym_server()
    print(f"Verifier: {verify_endpoint}")

    # --- Model: LoRA + vLLM fast_inference. Same recipe as the HF path but generation
    # runs on an in-process vLLM engine. offload_embedding intentionally dropped. ---
    from unsloth import FastLanguageModel
    max_seq_length = 768
    lora_rank = 32
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,            # in-process vLLM engine
        max_lora_rank=lora_rank,        # vLLM must know the max adapter rank up front
        gpu_memory_utilization=0.85,    # standby time-shares this with the optimizer step
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # --- Dataset ---
    from datasets import Dataset
    ds_path = f"{GYM_DIR}/resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl"
    train_data, max_len_seen = [], 0
    with open(ds_path) as f:
        for line in f:
            d = json.loads(line)
            task_prompt = d["responses_create_params"]["input"][0]["content"]
            train_data.append({
                "prompt": [{"role": "user", "content": task_prompt}],
                "answer": d["answer"],
                "metadata": d["metadata"],
            })
            plen = len(tokenizer.apply_chat_template(
                [{"role": "user", "content": task_prompt}], add_generation_prompt=True))
            max_len_seen = max(max_len_seen, plen)
    print(f"Loaded {len(train_data)} examples (max prompt len {max_len_seen}).")
    train_dataset = Dataset.from_list(train_data)

    # --- Reward: POST each rollout to the NeMo Gym deterministic verifier ---
    def reward_fn(completions, prompts=None, **kwargs):
        answers, metadatas = kwargs["answer"], kwargs["metadata"]
        scores = []
        for i, completion in enumerate(completions):
            text = completion[0]["content"]
            task_prompt = prompts[i][0]["content"]
            req = {
                "responses_create_params": {"input": [
                    {"role": "user", "content": task_prompt, "type": "message"}]},
                "response": {
                    "id": "resp", "created_at": 0, "model": MODEL_NAME, "object": "response",
                    "output": [{"id": "msg", "role": "assistant", "type": "message",
                                "status": "completed", "content": [
                                    {"type": "output_text", "text": text, "annotations": []}]}],
                    "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                },
                "question": task_prompt, "answer": answers[i], "metadata": metadatas[i],
            }
            try:
                r = requests.post(verify_endpoint, json=req, timeout=30)
                reward = r.json().get("reward", 0.0) if r.status_code == 200 else 0.0
            except requests.exceptions.RequestException as e:
                print(f"verify failed: {e}")
                reward = 0.0
            scores.append(reward)
        return np.array(scores)

    # --- GRPO config with vLLM colocate ---
    from trl import GRPOConfig, GRPOTrainer
    max_prompt_length = max_len_seen + 1
    args = GRPOConfig(
        temperature=1.0, learning_rate=1e-5, weight_decay=0.001, warmup_ratio=0.0,
        lr_scheduler_type="linear", optim="adamw_8bit", logging_steps=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=16, num_generations=8,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=1, max_steps=max_steps, save_steps=max_steps,
        report_to="wandb", run_name=f"rlvr-sudoku-{run_name}",
        output_dir="/outputs/checkpoints-vllm",
        epsilon_high=0.28, mask_truncated_completions=True,
        use_vllm=True, vllm_mode="colocate",   # <-- the vLLM switch
    )
    os.environ.setdefault("WANDB_PROJECT", "rlvr-sudoku")

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[reward_fn], args=args, train_dataset=train_dataset,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    vram = {
        "run": run_name,
        "backend": "vllm-colocate",
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL_NAME, "max_steps": max_steps,
    }
    os.makedirs("/outputs/metrics", exist_ok=True)
    with open(f"/outputs/metrics/vram_{run_name}.json", "w") as fp:
        json.dump(vram, fp, indent=2)
    print("VRAM:", vram)

    from huggingface_hub import HfApi
    hf_token = os.environ["HF_TOKEN"]
    if smoke:
        api = HfApi(token=hf_token)
        api.create_repo(hub_repo, private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=json.dumps(vram).encode(),
            path_in_repo="smoke_vram.json", repo_id=hub_repo,
        )
        print(f"Smoke HF push OK -> {hub_repo}/smoke_vram.json")
    else:
        model.push_to_hub_merged(hub_repo, tokenizer, save_method="lora", token=hf_token)
        print(f"Pushed LoRA adapter -> {hub_repo}")

    outputs.commit()
    return vram


@app.local_entrypoint()
def main(smoke: bool = False):
    result = train.remote(smoke=smoke)
    print("Done:", result)
