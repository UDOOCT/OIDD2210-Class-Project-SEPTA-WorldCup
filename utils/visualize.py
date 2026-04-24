"""
utils/visualize.py
------------------
Visualization helpers for SEPTA optimization results.

Functions:
  plot_demand_curve      — per-line demand across 61 slots
  plot_allocation_heatmap — train counts (line × slot)
  plot_fare_profile      — fare time series for one or all lines
  plot_profit_convergence — bilevel iteration history
  save_or_show           — unified save/display helper
"""

from __future__ import annotations
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _require_mpl():
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for visualizations. pip install matplotlib")


def save_or_show(fig, path: str | None = None):
    """Save figure to path (creates parent dirs) or show interactively."""
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
    else:
        plt.show()


# ── Slot labels (shared across plots) ────────────────────────────────────────
def _slot_labels():
    from data.parameters import TIME_SLOTS, slot_label
    return [slot_label(t) if i % 4 == 0 else "" for i, t in enumerate(TIME_SLOTS)]


def plot_demand_curve(demand: dict, lines: list | None = None,
                      worldcup: bool = True, save_path: str | None = None):
    """
    Line chart: demand per 15-min slot for selected lines.

    Args:
        demand:    dict[line] → np.array(61,)
        lines:     subset of lines to plot (None = all)
        worldcup:  whether demand includes World Cup overlay (affects title)
        save_path: file path to save; None to display interactively
    """
    _require_mpl()
    lines = lines or list(demand.keys())
    n = len(lines)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows),
                             sharex=True, sharey=False)
    axes = np.array(axes).flatten()
    labels = _slot_labels()

    for i, l in enumerate(lines):
        ax = axes[i]
        ax.fill_between(range(61), demand[l], alpha=0.4, color="steelblue")
        ax.plot(demand[l], color="steelblue", linewidth=1.2)
        ax.set_title(l, fontsize=9, fontweight="bold")
        ax.set_xticks(range(61))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_ylabel("Passengers", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    suffix = " (+ World Cup)" if worldcup else ""
    fig.suptitle(f"SEPTA RR Demand per 15-min Slot{suffix}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig


def plot_allocation_heatmap(result: dict, save_path: str | None = None):
    """
    Heatmap: optimal train counts across lines (rows) × time slots (cols).
    """
    _require_mpl()
    from data.parameters import TIME_SLOTS, slot_label

    lnames = list(result["lines"].keys())
    n_lines = len(lnames)
    matrix = np.array([result["lines"][l]["f"] for l in lnames])  # (n_lines, 61)

    fig, ax = plt.subplots(figsize=(18, max(4, n_lines * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=matrix.max() or 1)

    ax.set_yticks(range(n_lines))
    ax.set_yticklabels(lnames, fontsize=8)
    tick_pos = [i for i in range(61) if i % 4 == 0]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([slot_label(TIME_SLOTS[i]) for i in tick_pos],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Time slot (15 min)")
    ax.set_title(f"Train Allocation — profit=${result['profit']:,.0f}  pax={result['total_pax']:,.0f}",
                 fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Trains dispatched")
    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig


def plot_fare_profile(result: dict, lines: list | None = None,
                      save_path: str | None = None):
    """
    Line chart: optimal fare over time for selected lines.
    """
    _require_mpl()
    lines = lines or list(result["lines"].keys())
    labels = _slot_labels()

    fig, ax = plt.subplots(figsize=(14, 5))
    for l in lines:
        ax.plot(result["lines"][l]["p"], label=l, linewidth=1.4)

    ax.set_xticks(range(61))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fare ($)")
    ax.set_title("Optimal Fare Profile by Line", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig


def plot_profit_convergence(iterations: list, save_path: str | None = None):
    """
    Plot bilevel solver convergence: profit and Δpax per iteration.

    Args:
        iterations: list of dicts from run_bilevel result['iterations']
    """
    _require_mpl()
    iters  = [r["iter"]   for r in iterations]
    deltas = [r["delta"]  for r in iterations]
    profits= [r["profit"] for r in iterations]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(iters, profits, "o-", color="steelblue", linewidth=1.5)
    ax1.set_ylabel("Profit ($)")
    ax1.set_title("Bilevel Convergence", fontsize=11, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(linestyle="--", alpha=0.4)

    ax2.semilogy(iters, deltas, "o-", color="tomato", linewidth=1.5)
    ax2.set_ylabel("Δ pax (log scale)")
    ax2.set_xlabel("Iteration")
    ax2.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig
