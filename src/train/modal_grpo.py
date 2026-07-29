"""
RLVR Sudoku GRPO training on Modal.

Ported from unslothai/notebooks `NeMo-Gym-Sudoku.ipynb`. Trains Qwen2.5-1.5B-Instruct
on 4x4 mini-Sudoku via GRPO, with the deterministic reward computed by a NeMo Gym
resources server (reasoning_gym) running locally in the container.

The NeMo Gym setup (clone + uv venv + dataset) is baked into the image at build time so
it is cached and cold starts are fast. The verifier server (`ng_run`) is started at
runtime as a subprocess; the training loop POSTs each rollout to its /verify endpoint.

Usage:
    modal run src/train/modal_grpo.py --smoke          # 5-step gate (~$0.25)
    modal run src/train/modal_grpo.py                   # full 100-step baseline
"""

import modal

APP_NAME = "rlvr-sudoku-grpo"
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
GYM_DIR = "/root/Gym"
HF_REPO = "Samanthkoduru/rlvr-sudoku-grpo"          # full-run adapter target
HF_SMOKE_REPO = "Samanthkoduru/rlvr-sudoku-smoke"   # smoke-run push probe

app = modal.App(APP_NAME)

# Persistent volume for outputs (VRAM logs, checkpoints, metrics)
outputs = modal.Volume.from_name("rlvr-outputs", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: Unsloth training stack + NeMo Gym verifier baked in at build time.
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
    # Clone NeMo Gym and set up its isolated uv venv (python 3.12)
    .run_commands(
        f"git clone https://github.com/NVIDIA-NeMo/Gym.git {GYM_DIR}",
        f"cd {GYM_DIR} && uv venv --python 3.12",
        f"cd {GYM_DIR} && bash -c 'source .venv/bin/activate && uv sync'",
        f"cd {GYM_DIR} && bash -c 'source .venv/bin/activate && uv pip install reasoning-gym matplotlib'",
    )
    # Pre-generate the 2000-example mini_sudoku dataset (deterministic, seed 42)
    .run_commands(
        f"cd {GYM_DIR} && bash -c 'source .venv/bin/activate && python "
        "resources_servers/reasoning_gym/scripts/create_dataset.py "
        "--task mini_sudoku --size 2000 --seed 42 "
        "--output resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl'"
    )
    # --- xformers CUDA-build fix (appended last to preserve all cached layers) ---
    # pip pulls the cu128 build of xformers from PyPI while torch is 2.13.0+cu130,
    # so the CUDA extension can't load and Unsloth falls back to SDPA (slow + OOM).
    # Reinstall ONLY xformers from the matching cu130 index; --no-deps keeps torch
    # and everything else untouched. Then print xformers.info so the build log
    # confirms the CUDA extension loaded before we spend GPU time.
    .run_commands(
        "pip install --force-reinstall --no-deps "
        "--index-url https://download.pytorch.org/whl/cu130 xformers==0.0.35",
        "python -m xformers.info || true",
    )
    # expandable_segments cuts allocator fragmentation (the OOM traceback recommended it);
    # combined with the smaller seq/batch it keeps peak well under the L4's 22GB.
    .env({
        "UNSLOTH_VLLM_STANDBY": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)


def _start_nemo_gym_server():
    """Start the NeMo Gym resources server and return the /verify endpoint URL."""
    import os
    import subprocess
    import time
    import requests
    import yaml
    from omegaconf import OmegaConf

    head_port = 11000
    # Start ng_run head + resources server in the Gym venv
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

    # Wait for head server
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

    # Resolve resources server port from the head config
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
    timeout=18 * 60 * 60,  # hard cap: 18h so nothing drains all credits
    retries=0,             # never auto-retry an expensive run
)
def train(smoke: bool = False):
    import os
    import json
    import numpy as np
    import requests
    import torch

    max_steps = 5 if smoke else 100
    run_name = "smoke-5step" if smoke else "baseline-100step"
    hub_repo = HF_SMOKE_REPO if smoke else HF_REPO
    print(f"=== RLVR Sudoku GRPO | run={run_name} | max_steps={max_steps} ===")

    verify_endpoint = _start_nemo_gym_server()
    print(f"Verifier: {verify_endpoint}")

    # --- Model: LoRA + gradient checkpointing (memory-efficient path; fits L4 24GB).
    # Full finetune at seq 4096 x 8 generations OOMs a 22GB L4 — smoke test caught this.
    # Seq 2048 also OOMed: scoring 512 completions (8 gen x 64 accum) at 2048 tok blew
    # the 24GB L4 in the GRPO ref-logprob forward. 4x4 Sudoku prompts are <=147 tok and a
    # solved grid is ~40 tok, so 768 is ample (prompt 148 + ~620 completion). ---
    from unsloth import FastLanguageModel
    max_seq_length = 768  # was 2048; Sudoku needs far less, and it caps GRPO scoring memory
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        full_finetuning=False,
        offload_embedding=True,
    )
    lora_rank = 32
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

    # --- GRPO config (notebook values; W&B on) ---
    from trl import GRPOConfig, GRPOTrainer
    max_prompt_length = max_len_seen + 1
    args = GRPOConfig(
        temperature=1.0, learning_rate=1e-5, weight_decay=0.001, warmup_ratio=0.0,
        lr_scheduler_type="linear", optim="adamw_8bit", logging_steps=1,
        # grad_accum 16 (was 64): GRPO generates & scores pdbs*accum*... = 128 completions
        # per round instead of 512 → ~4x less peak memory and ~4x faster generation. The
        # per-prompt group is still num_generations=8, so the advantage estimate is unchanged.
        per_device_train_batch_size=1, gradient_accumulation_steps=16, num_generations=8,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=1, max_steps=max_steps, save_steps=max_steps,
        report_to="wandb", run_name=f"rlvr-sudoku-{run_name}",
        output_dir="/outputs/checkpoints",
        epsilon_high=0.28, mask_truncated_completions=True,
    )
    os.environ.setdefault("WANDB_PROJECT", "rlvr-sudoku")

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[reward_fn], args=args, train_dataset=train_dataset,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    # --- Capture VRAM (Artifact 1 data) ---
    vram = {
        "run": run_name,
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL_NAME, "max_steps": max_steps,
    }
    os.makedirs("/outputs/metrics", exist_ok=True)
    with open(f"/outputs/metrics/vram_{run_name}.json", "w") as fp:
        json.dump(vram, fp, indent=2)
    print("VRAM:", vram)

    # --- Push (smoke: probe HF write with a tiny upload; full: merged 16bit) ---
    from huggingface_hub import HfApi
    hf_token = os.environ["HF_TOKEN"]
    if smoke:
        api = HfApi(token=hf_token)
        api.create_repo(hub_repo, private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=json.dumps(vram).encode(),
            path_in_repo="smoke_vram.json", repo_id=hub_repo,
        )
        print(f"Smoke HF push OK → {hub_repo}/smoke_vram.json")
    else:
        # LoRA adapter is tiny → fast push; keep the effort-knob adapters comparable
        model.push_to_hub_merged(hub_repo, tokenizer, save_method="lora", token=hf_token)
        print(f"Pushed LoRA adapter → {hub_repo}")

    outputs.commit()
    return vram


@app.local_entrypoint()
def main(smoke: bool = False):
    result = train.remote(smoke=smoke)
    print("Done:", result)
