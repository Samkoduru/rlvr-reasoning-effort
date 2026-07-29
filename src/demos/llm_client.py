"""
Shared Nemotron client for the Phase-3 live demos (Examples 1 & 3).

Wraps the NVIDIA Build OpenAI-compatible endpoint and captures the two things the blog
demos measure: the reasoning-effort toggle ("detailed thinking on/off") and per-call
token spend + latency. Nemotron returns the chain-of-thought in a separate
`reasoning_content` field (NOT inline <think> tokens) — verified live 2026-07-27 — so we
read both `content` and `reasoning_content` and count reasoning tokens explicitly.

Env: needs NVIDIA_API_KEY. Either export it, or call
`from src.utils.secrets import load_secrets; load_secrets()` first to pull it from AWS SM.

This module is pure inference against the hosted API — no GPU, no training. Example 2
(budget forcing) deliberately does NOT use this client: forcing </think> mid-generation
needs a model whose decode loop we own, so it runs against a local/served model instead.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict

from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Nemotron reasoning model with a CLEAN effort toggle. Verified live 2026-07-27:
#   off -> 0 reasoning chars / 3 completion tokens (direct answer)
#   on  -> ~1100 reasoning chars / ~370 tokens (full CoT), same correct answer.
# NB: the "-v1.5" refresh reasons regardless of the toggle (off still ~2900 chars), and the
# nano-9b-v2 doesn't suppress either — v1 is the one that honors "detailed thinking off".
DEFAULT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

# The reasoning-effort knob is a system-prompt toggle, not a decode parameter.
EFFORT_ON = "detailed thinking on"
EFFORT_OFF = "detailed thinking off"


@dataclass
class LLMResult:
    """One Nemotron call, with everything the artifacts need."""
    model: str
    effort: str                       # "on" | "off"
    prompt: str
    content: str                      # the final answer
    reasoning_content: str            # the CoT trace (empty when effort is off)
    prompt_tokens: int
    completion_tokens: int            # includes reasoning tokens the API bills as completion
    reasoning_chars: int              # len(reasoning_content) — a proxy when the API omits a token count
    total_tokens: int
    latency_s: float
    finish_reason: str
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flat dict for CSV / DataFrame rows feeding the artifacts."""
        return asdict(self)


def _client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Export it, or run "
            "`from src.utils.secrets import load_secrets; load_secrets()` first."
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def call(
    prompt: str,
    effort: str = "on",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    system_extra: str | None = None,
) -> LLMResult:
    """
    One Nemotron completion with the reasoning-effort toggle.

    effort="on"  -> system prompt "detailed thinking on"  (model emits reasoning_content)
    effort="off" -> system prompt "detailed thinking off" (terse, little/no reasoning)

    Returns an LLMResult with token counts + latency for the compute-vs-difficulty artifacts.
    """
    if effort not in ("on", "off"):
        raise ValueError("effort must be 'on' or 'off'")
    system = EFFORT_ON if effort == "on" else EFFORT_OFF
    if system_extra:
        system = f"{system}\n{system_extra}"

    client = _client()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - t0

    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
    # reasoning_content is a Nemotron/OpenAI-extension field; not on every SDK model object.
    reasoning = getattr(msg, "reasoning_content", None) or ""
    usage = resp.usage

    return LLMResult(
        model=model,
        effort=effort,
        prompt=prompt,
        content=content,
        reasoning_content=reasoning,
        prompt_tokens=getattr(usage, "prompt_tokens", 0),
        completion_tokens=getattr(usage, "completion_tokens", 0),
        reasoning_chars=len(reasoning),
        total_tokens=getattr(usage, "total_tokens", 0),
        latency_s=round(latency, 3),
        finish_reason=choice.finish_reason or "",
    )


if __name__ == "__main__":
    # Smoke check: needs NVIDIA_API_KEY (or AWS SM creds). Compares effort on vs off.
    try:
        from src.utils.secrets import load_secrets
        load_secrets()
    except Exception as e:  # noqa: BLE001 - best-effort; env var may already be set
        print(f"(secrets load skipped: {e})")

    q = "A farmer has 17 sheep. All but 9 run away. How many are left? Answer with just the number."
    for eff in ("off", "on"):
        r = call(q, effort=eff, max_tokens=1024)
        print(f"\n=== effort {eff} ===")
        print(f"answer: {r.content.strip()[:120]}")
        print(f"reasoning_chars={r.reasoning_chars} completion_tokens={r.completion_tokens} "
              f"total_tokens={r.total_tokens} latency={r.latency_s}s")
