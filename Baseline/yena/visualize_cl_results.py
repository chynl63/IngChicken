"""
Visualize Continual Learning results across all 4 LIBERO benchmarks.

Usage:
    cd /home/kyn7666/cl_diffusion_libero-object
    python visualize_cl_results.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_ROOT = Path("/home/kyn7666/cl_diffusion_libero-object/results")
OUT_DIR = Path("/home/kyn7666/cl_diffusion_libero-object/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = {
    "libero_object": "LIBERO-Object",
    "libero_spatial": "LIBERO-Spatial",
    "libero_goal":    "LIBERO-Goal",
    "libero_long":    "LIBERO-Long",
}

COLORS = {
    "libero_object":  "#4C72B0",
    "libero_spatial": "#DD8452",
    "libero_goal":    "#55A868",
    "libero_long":    "#C44E52",
}

# ── helpers ────────────────────────────────────────────────────────────────

def compute_nbt(mat):
    """NBT = mean over j of (max_i R[i,j] - R[N-1,j])  for j < N-1."""
    N = mat.shape[0]
    total, count = 0.0, 0
    for j in range(N - 1):
        valid = [mat[i, j] for i in range(j, N) if not np.isnan(mat[i, j])]
        if len(valid) < 2:
            continue
        final = mat[N - 1, j]
        if not np.isnan(final):
            total += max(valid) - final
            count += 1
    return total / max(count, 1)

def short_name(name):
    """Shorten a task name for axis labels."""
    # keep last meaningful segment after last underscore-word boundary
    parts = name.split("_")
    # take up to 4 words from the end
    return " ".join(parts[-4:]) if len(parts) >= 4 else name.replace("_", " ")


def build_from_eval_log(log, task_names_hint=None):
    """Reconstruct performance matrix from eval_log.json entries."""
    # Collect all task names in order
    ordered_names = []
    for entry in log:
        tn = entry["task_name"]
        if tn not in ordered_names:
            ordered_names.append(tn)

    # N = highest task index seen + 1
    max_idx = max(entry["task_idx"] for entry in log)
    N = max_idx + 1

    # Use hint for full task list if available (entries may only cover 0..N-2)
    if task_names_hint and len(task_names_hint) >= N:
        task_names = task_names_hint
    else:
        task_names = ordered_names[:N] if len(ordered_names) >= N else \
            ordered_names + [f"task_{i}" for i in range(len(ordered_names), N)]

    mat = np.full((N, N), np.nan)
    for entry in log:
        i = entry["task_idx"]
        for j_str, res in entry["task_results"].items():
            j = int(j_str)
            if j < N:
                mat[i, j] = res["success_rate"]

    avg_sr = [float(np.nanmean(mat[i, :i+1])) for i in range(N)]

    # forgetting[j] = diagonal SR - last-row SR for task j
    forgetting = []
    for j in range(N):
        peak = mat[j, j]
        last_eval = mat[max_idx, j]
        if not np.isnan(peak) and not np.isnan(last_eval):
            forgetting.append(float(peak - last_eval))
        else:
            forgetting.append(float("nan"))

    nbt = compute_nbt(mat)
    final_sr = float(np.nanmean(mat[max_idx, :max_idx+1]))

    return dict(task_names=task_names, mat=mat, avg_sr=avg_sr,
                forgetting=forgetting, nbt=nbt, final_sr=final_sr)


def load_results(key):
    """
    Returns dict with keys:
        task_names, mat (N×N ndarray), avg_sr, forgetting, nbt, final_sr
    mat[i, j] = SR on task j after training task i  (NaN if not evaluated)
    """
    results_dir = RESULTS_ROOT / f"cl_{key}"

    # Try final results.json first (has real numeric values)
    p = results_dir / "results.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        task_names = d["task_names"]
        N = len(task_names)
        pm = d.get("performance_matrix", {})
        mat = np.full((N, N), np.nan)
        for i, sk in enumerate(sorted(pm.keys())):
            for j, tname in enumerate(task_names):
                v = pm[sk].get(tname)
                if v is not None:
                    mat[i, j] = float(v)
        avg_sr = [float(np.nanmean(mat[i, :i+1])) for i in range(N)]
        forgetting = []
        for j in range(N):
            peak, last = mat[j, j], mat[N-1, j]
            forgetting.append(float(peak - last) if not np.isnan(peak) and not np.isnan(last) else float("nan"))
        nbt = compute_nbt(mat)
        final_sr = float(np.nanmean(mat[N-1]))
        return dict(task_names=task_names, mat=mat, avg_sr=avg_sr,
                    forgetting=forgetting, nbt=nbt, final_sr=final_sr)

    # Fall back to eval_log.json (spatial/goal have None-valued intermediates)
    log_p = results_dir / "eval_log.json"
    if log_p.exists():
        with open(log_p) as f:
            log = json.load(f)
        # Use task_names from intermediate if available
        hint = None
        inter_p = results_dir / "results_intermediate.json"
        if inter_p.exists():
            with open(inter_p) as f:
                hint = json.load(f).get("task_names")
        return build_from_eval_log(log, task_names_hint=hint)

    raise FileNotFoundError(f"No result files found in {results_dir}")


# ── Figure 1: 4 heatmaps ───────────────────────────────────────────────────

def plot_heatmaps(data):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Performance Matrix — Continual Learning with Diffusion Policy\n"
                 "(row = after training task i, col = evaluated on task j)",
                 fontsize=14, fontweight="bold", y=0.98)

    for ax, (key, title) in zip(axes.flat, BENCHMARKS.items()):
        d = data[key]
        mat = d["mat"]
        N = len(d["task_names"])
        labels = [short_name(n) for n in d["task_names"]]

        nrows, ncols = mat.shape
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Evaluated Task", fontsize=10)
        ax.set_ylabel("Trained Through Task", fontsize=10)
        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.set_xticklabels([f"T{i}" for i in range(ncols)], fontsize=9)
        ax.set_yticklabels([f"After T{i}" for i in range(nrows)], fontsize=9)

        # annotate cells
        for i in range(nrows):
            for j in range(ncols):
                if not np.isnan(mat[i, j]) and j <= i:
                    val = mat[i, j]
                    color = "black" if 0.3 < val < 0.75 else "white"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Success Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "heatmaps_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: avg SR over training stages ─────────────────────────────────

def plot_avg_sr(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Average Success Rate per Training Stage",
                 fontsize=13, fontweight="bold")

    for key, title in BENCHMARKS.items():
        d = data[key]
        x = list(range(1, len(d["avg_sr"]) + 1))
        ax.plot(x, d["avg_sr"], marker="o", label=title, color=COLORS[key], linewidth=2)

    ax.set_xlabel("Number of Tasks Trained", fontsize=11)
    ax.set_ylabel("Avg Success Rate (seen tasks)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = OUT_DIR / "avg_sr_over_stages.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: forgetting per task (grouped bar) ───────────────────────────

def plot_forgetting(data):
    N = 10
    x = np.arange(N)
    n_bm = len(BENCHMARKS)
    width = 0.2
    offsets = np.linspace(-(n_bm-1)/2 * width, (n_bm-1)/2 * width, n_bm)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Catastrophic Forgetting per Task\n"
                 "(SR right after learning — SR at end of all training)",
                 fontsize=12, fontweight="bold")

    for idx, (key, title) in enumerate(BENCHMARKS.items()):
        d = data[key]
        vals = [d["forgetting"][i] if i < len(d["forgetting"]) else float("nan") for i in range(N)]
        bars = ax.bar(x + offsets[idx], vals, width, label=title,
                      color=COLORS[key], alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Task Index", fontsize=11)
    ax.set_ylabel("Forgetting (↑ = more forgetting)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(N)])
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    out = OUT_DIR / "forgetting_per_task.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 4: cross-benchmark summary ─────────────────────────────────────

def plot_summary(data):
    keys = list(BENCHMARKS.keys())
    titles = list(BENCHMARKS.values())
    colors = [COLORS[k] for k in keys]

    nbt_vals    = [data[k]["nbt"]      for k in keys]
    final_vals  = [data[k]["final_sr"] for k in keys]
    avg_forgetting = [np.nanmean(data[k]["forgetting"]) for k in keys]

    x = np.arange(len(keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Cross-Benchmark Summary", fontsize=13, fontweight="bold")

    b1 = ax.bar(x - width, nbt_vals,    width, label="NBT (avg. peak SR)",    color=[c+"bb" for c in colors], edgecolor="white")
    b2 = ax.bar(x,         final_vals,  width, label="Final Avg SR",           color=colors, edgecolor="white")
    b3 = ax.bar(x + width, avg_forgetting, width, label="Avg Forgetting",     color=[c+"66" for c in colors], edgecolor="white", hatch="//")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    out = OUT_DIR / "summary_cross_benchmark.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 5: diagonal (just-learned SR) per benchmark ────────────────────

def plot_diagonal(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Success Rate Right After Learning Each Task (Diagonal)",
                 fontsize=12, fontweight="bold")

    for key, title in BENCHMARKS.items():
        d = data[key]
        mat = d["mat"]
        Nmat = mat.shape[0]
        diag = [mat[i, i] if not np.isnan(mat[i, i]) else np.nan for i in range(Nmat)]
        ax.plot(range(1, Nmat+1), diag, marker="s", label=title,
                color=COLORS[key], linewidth=2, markersize=7)

    ax.set_xlabel("Task Index (training order)", fontsize=11)
    ax.set_ylabel("Success Rate on Just-Learned Task", fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"T{i}" for i in range(10)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = OUT_DIR / "diagonal_sr.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...")
    data = {}
    for key in BENCHMARKS:
        try:
            data[key] = load_results(key)
            d = data[key]
            print(f"  {BENCHMARKS[key]:18s}  NBT={d['nbt']:.3f}  final_SR={d['final_sr']:.3f}")
        except Exception as e:
            print(f"  {key}: ERROR — {e}")

    print("\nGenerating figures...")
    plot_heatmaps(data)
    plot_avg_sr(data)
    plot_forgetting(data)
    plot_summary(data)
    plot_diagonal(data)

    print(f"\nAll figures saved to: {OUT_DIR}/")
