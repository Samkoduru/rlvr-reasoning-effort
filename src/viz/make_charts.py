"""
Editorial data-viz for the RLVR reasoning-effort post (personal — Medium / LinkedIn).

Modern light theme, validated categorical palette (indigo/teal/amber/rose, all CVD-safe),
rich multi-panel figures built from the full Modal + WandB + Nemotron telemetry.

Outputs -> blog/figures/*.png  (2x DPI for retina/LinkedIn)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec

# --- palette (validated) ---
BG = "#FFFFFF"
PANEL = "#F8FAFC"
INK = "#1E293B"      # slate-800  primary text
TITLE = "#0F172A"    # slate-900  titles
MUTED = "#64748B"    # slate-500  secondary
GRID = "#E7ECF2"
INDIGO = "#4F46E5"
TEAL = "#0D9488"
AMBER = "#D97706"
ROSE = "#E11D48"
EMERALD = "#059669"
OK = "#10B981"
NO = "#F1A0AE"       # muted rose for "incorrect" cells

FIG_DIR = "blog/figures"
DATA = "data"

for _cand in ["Helvetica Neue", "Helvetica", "Arial"]:
    try:
        font_manager.findfont(_cand, fallback_to_default=False)
        FONT = _cand
        break
    except Exception:
        FONT = "DejaVu Sans"


def apply_style():
    plt.rcParams.update({
        "font.family": [FONT, "DejaVu Sans"],  # DejaVu supplies the → glyph Helvetica lacks
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TITLE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    })


def _clean(ax, xgrid=False):
    ax.tick_params(length=0, labelsize=10)
    ax.grid(axis="x", visible=xgrid)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    p = f"{FIG_DIR}/{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("wrote", p)


def _smooth(y, k=5):
    return pd.Series(y).rolling(k, center=True, min_periods=1).mean().values


# ---------------------------------------------------------------------------
# 1. Training dynamics (hero) — reward + KL + completion length + confidence.
# ---------------------------------------------------------------------------
def training_dynamics():
    d = pd.read_csv(f"{DATA}/metrics/training_dynamics.csv")
    s = d["step"]
    fig = plt.figure(figsize=(10, 7.2))
    gs = GridSpec(2, 3, height_ratios=[1.5, 1], hspace=0.42, wspace=0.32,
                  left=0.08, right=0.96, top=0.86, bottom=0.09)

    # --- hero: reward ---
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(s, d.reward - d.reward_std, d.reward + d.reward_std,
                    color=INDIGO, alpha=0.10, linewidth=0)
    ax.plot(s, d.reward, color=INDIGO, alpha=0.25, linewidth=1)
    ax.plot(s, _smooth(d.reward), color=INDIGO, linewidth=2.8)
    ax.scatter([s.iloc[0]], [d.reward.iloc[0]], color=INDIGO, s=45, zorder=5)
    imax = d.reward.idxmax()
    ax.scatter([s[imax]], [d.reward[imax]], color=AMBER, s=60, zorder=5)
    ax.annotate("0.09", (s.iloc[0], d.reward.iloc[0]), xytext=(8, -14),
                textcoords="offset points", color=INK, fontsize=11)
    ax.annotate(f"peak {d.reward[imax]:.2f}", (s[imax], d.reward[imax]), xytext=(-6, 12),
                textcoords="offset points", color=AMBER, fontsize=11)
    ax.axhspan(0.47, 0.55, color=TEAL, alpha=0.05)
    ax.annotate("plateau ~0.50", (34, 0.575), color=TEAL, fontsize=10)
    ax.set_title("Verifier reward — the model learns to solve Sudoku from reward alone",
                 fontsize=13, loc="left", pad=8)
    ax.set_ylabel("reward")
    ax.set_xlim(0, 100); ax.set_ylim(0, 0.62)
    _clean(ax)

    def mini(col, data, color, title, ylab):
        a = fig.add_subplot(gs[1, col])
        a.plot(s, _smooth(data), color=color, linewidth=2.2)
        a.plot(s, data, color=color, alpha=0.18, linewidth=0.8)
        a.set_title(title, fontsize=11, loc="left", color=INK, pad=6)
        a.set_xlabel("step", fontsize=9)
        a.set_ylabel(ylab, fontsize=9)
        a.set_xlim(0, 100)
        _clean(a)
        return a

    mini(0, d.kl, TEAL, "KL divergence", "kl")
    mini(1, d.completion_length, AMBER, "Completion length", "tokens")
    mini(2, d.reward_std, ROSE, "Reward spread (↓ = surer)", "std")

    fig.suptitle("Watching reasoning get trained in — 100 GRPO steps, one L4, ~76 min",
                 fontsize=16, color=TITLE, x=0.08, ha="left", y=0.95)
    save(fig, "01_training_dynamics")


# ---------------------------------------------------------------------------
# 2. Efficiency — LoRA trainable share + VRAM headroom.
# ---------------------------------------------------------------------------
def efficiency():
    base = json.load(open(f"{DATA}/metrics/vram_baseline-100step.json"))
    smoke = json.load(open(f"{DATA}/metrics/vram_smoke-5step.json"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3),
                                 gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1.25]))

    # trainable share
    trainable = 36.9 / 1580.6 * 100
    a1.barh([0], [100], color=GRID, height=0.5)
    a1.barh([0], [trainable], color=INDIGO, height=0.5)
    a1.set_xlim(0, 100); a1.set_ylim(-0.6, 0.6)
    a1.set_yticks([])
    a1.set_title("Only 2.3% of the model is trained", fontsize=12, loc="left", color=TITLE, pad=8)
    a1.annotate("36.9M LoRA params", (trainable, 0), xytext=(10, 22),
                textcoords="offset points", color=INDIGO, fontsize=11)
    a1.annotate("1.58B frozen", (100, 0), xytext=(-8, -30), ha="right",
                textcoords="offset points", color=MUTED, fontsize=11)
    a1.set_xlabel("% of parameters")
    a1.grid(False)
    for sp in a1.spines.values():
        sp.set_visible(False)
    a1.tick_params(length=0, labelsize=9)

    # VRAM
    labels = ["smoke\nalloc", "run\nalloc", "run\nreserved"]
    vals = [smoke["max_allocated_gb"], base["max_allocated_gb"], base["max_reserved_gb"]]
    bars = a2.bar(labels, vals, color=[TEAL, INDIGO, INDIGO], width=0.62, zorder=3)
    a2.axhline(22.0, color=ROSE, linewidth=1.6, linestyle=(0, (5, 3)))
    a2.annotate("L4 usable ~22 GB", (2.4, 22), ha="right", va="bottom", color=ROSE, fontsize=10)
    a2.annotate("full fine-tune → OOM", (2.4, 24.3), ha="right", color=MUTED, fontsize=9.5)
    for b, v in zip(bars, vals):
        a2.annotate(f"{v:.1f}", (b.get_x()+b.get_width()/2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK, fontsize=11)
    a2.set_ylim(0, 25); a2.set_ylabel("GB")
    a2.set_title("Peak VRAM fits a 24 GB L4 with headroom", fontsize=12, loc="left", color=TITLE, pad=8)
    _clean(a2)
    fig.suptitle("Why one cheap GPU was enough", fontsize=16, color=TITLE, x=0.06, ha="left", y=1.02)
    save(fig, "02_efficiency")


# ---------------------------------------------------------------------------
# 3. Effort modes (Example 1) — per-problem dumbbell of token cost.
# ---------------------------------------------------------------------------
def effort_modes():
    df = pd.read_csv(f"{DATA}/phase3/effort_modes.csv")
    piv = df.pivot_table(index="problem_id", columns="effort",
                         values=["completion_tokens", "correct"]).fillna(0)
    piv = piv.sort_values(("completion_tokens", "on"))
    y = np.arange(len(piv))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, pid in enumerate(piv.index):
        off_t = max(piv.loc[pid, ("completion_tokens", "off")], 1)
        on_t = piv.loc[pid, ("completion_tokens", "on")]
        ax.plot([off_t, on_t], [i, i], color=GRID, linewidth=2.5, zorder=1)
        ax.scatter([off_t], [i], color=MUTED, s=55, zorder=3)
        off_ok = piv.loc[pid, ("correct", "off")] >= 1
        ax.scatter([on_t], [i], color=INDIGO if off_ok else ROSE, s=90, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=10)
    ax.set_xlabel("completion tokens  (log scale)")
    ax.set_xlim(1, 1200)
    # legend below the plot (data occupies both sides on the log axis)
    ax.scatter([], [], color=MUTED, s=55, label="effort off")
    ax.scatter([], [], color=INDIGO, s=90, label="effort on — off already correct")
    ax.scatter([], [], color=ROSE, s=90, label="effort on — off was wrong")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              frameon=False, fontsize=9.5, labelcolor=INK)
    ax.set_title("Reasoning on demand: the same model, 2 tokens vs ~370",
                 fontsize=15, color=TITLE, loc="left", pad=30)
    ax.annotate("off: 80% correct, 2 tok avg      on: 100% correct, 372 tok avg",
                (0.0, 1.03), xycoords="axes fraction", color=MUTED, fontsize=10.5)
    _clean(ax, xgrid=True)
    ax.grid(axis="y", visible=False)
    save(fig, "03_effort_modes")


# ---------------------------------------------------------------------------
# 4. Budget forcing (Example 2) — correctness heatmap + marginal accuracy.
# ---------------------------------------------------------------------------
def budget_forcing():
    df = pd.read_csv(f"{DATA}/phase3/budget_forcing.csv")
    df["budget"] = df["budget"].astype(str)
    order = ["40", "100", "200", "400", "none"]
    probs = list(df["problem_id"].unique())
    M = np.zeros((len(probs), len(order)))
    T = np.zeros_like(M)
    for i, p in enumerate(probs):
        for j, b in enumerate(order):
            r = df[(df.problem_id == p) & (df.budget == b)]
            if len(r):
                M[i, j] = r.correct.iloc[0]
                T[i, j] = r.thinking_tokens_used.iloc[0]

    fig = plt.figure(figsize=(8.6, 5.6))
    gs = GridSpec(2, 1, height_ratios=[1, 3.2], hspace=0.08, left=0.16, right=0.97,
                  top=0.88, bottom=0.12)
    # marginal accuracy
    at = fig.add_subplot(gs[0])
    acc = M.mean(axis=0) * 100
    at.plot(range(len(order)), acc, color=INDIGO, linewidth=2.4, marker="o",
            markersize=7, markerfacecolor=INDIGO, zorder=3)
    for j, v in enumerate(acc):
        at.annotate(f"{v:.0f}%", (j, v), xytext=(0, 7), textcoords="offset points",
                    ha="center", color=INK, fontsize=10)
    at.set_xlim(-0.5, len(order)-0.5); at.set_ylim(-8, 108)
    at.set_xticks([]); at.set_ylabel("accuracy", fontsize=9)
    at.set_title("Budget forcing: how much thinking is enough?", fontsize=15,
                 color=TITLE, loc="left", pad=8)
    _clean(at)

    # heatmap
    ax = fig.add_subplot(gs[1])
    from matplotlib.colors import ListedColormap
    ax.imshow(M, cmap=ListedColormap([NO, OK]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["40", "100", "200", "400", "none"], fontsize=10)
    ax.set_yticks(range(len(probs))); ax.set_yticklabels(probs, fontsize=10)
    ax.set_xlabel("thinking-token budget  (</think> injected at cap)")
    for i in range(len(probs)):
        for j in range(len(order)):
            ax.text(j, i, "✓" if M[i, j] else "✗", ha="center", va="center",
                    color="white", fontsize=12)
    ax.set_xticks(np.arange(-.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(probs), 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(length=0)
    ax.grid(which="major", visible=False)
    save(fig, "04_budget_forcing")


# ---------------------------------------------------------------------------
# 5. Adaptive compute (Example 3) — reasoning tokens vs difficulty (size=latency).
# ---------------------------------------------------------------------------
def adaptive_compute():
    df = pd.read_csv(f"{DATA}/phase3/adaptive_compute.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    sizes = 40 + (df.latency_s / df.latency_s.max()) * 320
    ax.scatter(df.steps + np.random.RandomState(1).uniform(-0.06, 0.06, len(df)),
               df.completion_tokens, s=sizes, color=INDIGO, alpha=0.55,
               edgecolor="white", linewidth=1.2, zorder=3)
    means = df.groupby("steps").completion_tokens.mean()
    ax.plot(means.index, means.values, color=AMBER, linewidth=2.6, zorder=4,
            marker="D", markersize=8, markeredgecolor="white", markeredgewidth=1.4)
    ax.annotate("mean", (means.index[-1], means.values[-1]), xytext=(-42, 6),
                textcoords="offset points", color=AMBER, fontsize=11)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("required solution steps  (difficulty →)")
    ax.set_ylabel("reasoning tokens")
    ax.set_title("The model scales its own thinking to difficulty",
                 fontsize=15, color=TITLE, loc="left", pad=26)
    ax.annotate("marker size = latency", (0.0, 1.03), xycoords="axes fraction",
                color=MUTED, fontsize=10)
    _clean(ax)
    save(fig, "05_adaptive_compute")


# ---------------------------------------------------------------------------
# 6. Reward hacking — illustrative divergence (after arXiv:2505.22203).
# ---------------------------------------------------------------------------
def reward_hacking():
    it = np.linspace(0, 600, 160)
    reward = 0.15 + 0.8 / (1 + np.exp(-(it - 250) / 55))
    acc = 0.30 + 0.30 * np.exp(-((it - 175) / 120) ** 2) - 0.22 * (it > 300) * ((it - 300) / 300)
    acc = np.clip(acc, 0.05, 0.63)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(it, reward, color=AMBER, linewidth=3, zorder=3)
    ax.plot(it, acc, color=INDIGO, linewidth=3, zorder=4)
    ax.annotate("proxy reward", (600, reward[-1]), xytext=(-100, 4),
                textcoords="offset points", color=AMBER, fontsize=12)
    ax.annotate("true accuracy", (600, acc[-1]), xytext=(-108, -2),
                textcoords="offset points", color=INDIGO, fontsize=12)
    ax.axvline(300, color=MUTED, linewidth=1, alpha=0.5, linestyle=(0, (4, 3)))
    ax.annotate("reward hacking begins", (308, 0.60), color=MUTED, fontsize=10)
    ax.set_xlabel("training iterations"); ax.set_ylabel("score (normalized)")
    ax.set_ylim(0, 1)
    ax.set_title("When the reward is gameable, accuracy diverges",
                 fontsize=15, color=TITLE, loc="left", pad=10)
    ax.text(0, -0.19, "Illustrative, after Wang et al. (arXiv:2505.22203). Our deterministic-verifier run does not exhibit this.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    _clean(ax)
    save(fig, "06_reward_hacking")


# ---------------------------------------------------------------------------
# 7. Architecture — the RLVR loop.
# ---------------------------------------------------------------------------
def architecture():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    boxes = [
        ("Sudoku prompt", "4×4 puzzle", TEAL),
        ("Policy model", "Qwen2.5-1.5B + LoRA\nG rollouts", INDIGO),
        ("NeMo Gym verifier", "deterministic\nreward / rollout", TEAL),
        ("GRPO", "group-relative\nadvantages", INDIGO),
        ("LoRA update", "36.9M params\n2.3% of model", AMBER),
    ]
    bw, bh, y = 16.5, 30, 55
    cx = [10 + i * 20 for i in range(5)]
    for x, (t, sub, c) in zip(cx, boxes):
        ax.add_patch(FancyBboxPatch((x-bw/2, y-bh/2), bw, bh,
                     boxstyle="round,pad=0.5,rounding_size=2.4",
                     facecolor=c, edgecolor="none", zorder=3))
        ax.text(x, y+5, t, ha="center", va="center", color="white", fontsize=11.5, zorder=4)
        ax.text(x, y-6.5, sub, ha="center", va="center", color="white", fontsize=8.6,
                zorder=4, alpha=0.92)
    for a, b in zip(cx[:-1], cx[1:]):
        ax.add_patch(FancyArrowPatch((a+bw/2, y), (b-bw/2, y), arrowstyle="-|>",
                     mutation_scale=18, color=MUTED, lw=2, zorder=2))
    ax.add_patch(FancyArrowPatch((cx[-1], y-bh/2), (cx[1], y-bh/2),
                 connectionstyle="arc3,rad=-0.32", arrowstyle="-|>", mutation_scale=18,
                 color=AMBER, lw=2.2, zorder=1))
    ax.text((cx[1]+cx[-1])/2, y-bh/2-26,
            "policy improves each step   ·   reward 0.09 → 0.55 over 100 steps",
            ha="center", color=INK, fontsize=10.5)
    ax.set_title("The RLVR loop: rollouts → deterministic reward → group-relative update",
                 fontsize=15, color=TITLE, pad=2)
    save(fig, "07_architecture")


if __name__ == "__main__":
    apply_style()
    print("font:", FONT)
    training_dynamics()
    efficiency()
    effort_modes()
    budget_forcing()
    adaptive_compute()
    reward_hacking()
    architecture()
    print("done")
