"""
Post-simulation market share plotting utilities.

All functions accept DataFrames produced by :mod:`post_simulation.market_share`
and return a ``matplotlib.figure.Figure`` (and optionally save it to disk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _unique_introduce_from_steps(
    generator_introduce_from: Optional[Dict[str, int]],
) -> List[int]:
    """Return sorted unique introduce_from timesteps (excluding 0)."""
    if not generator_introduce_from:
        return []
    return sorted({t for t in generator_introduce_from.values() if t > 0})


def _mask_pre_introduction_display(
    display_df: pd.DataFrame,
    generator_introduce_from: Dict[str, int],
) -> pd.DataFrame:
    """Set NaN for display timesteps before each generator's introduce_from.

    display_df index: 0 = synthetic start, 1 = sim t=0, 2 = sim t=1, ...
    So simulation t = index - 1. We mask where sim_t < introduce_from.
    """
    out = display_df.copy()
    sim_t = out.index.values - 1
    for col in out.columns:
        intro = generator_introduce_from.get(col, 0)
        mask = (sim_t < intro) & ((sim_t >= 0) | (intro > 0))
        out.loc[mask, col] = np.nan
    return out


def _market_share_for_plot(market_share: pd.DataFrame) -> pd.DataFrame:
    """Prepend a row of zeros so every model starts at 0%, shift index to 1-based.

    Simulation timestep t (0, 1, ..., T-1) maps to display timestep (1, 2, ..., T).
    The x-axis runs 0, 1, 2, ..., T where 0 = start (all 0%) and 1..T = steps 1..T.
    """
    is_int_index = hasattr(market_share.index, "dtype") and (
        market_share.index.dtype.kind in "iu"
        or str(market_share.index.dtype).startswith("int")
    )
    if not is_int_index or market_share.empty:
        return market_share

    zero_row = pd.DataFrame(0.0, index=[0], columns=market_share.columns)
    shifted = market_share.copy()
    shifted.index = shifted.index + 1
    return pd.concat([zero_row, shifted]).sort_index()


def _window_labels_to_t_ends(
    market_share_windows: pd.DataFrame,
) -> Tuple[List[int], List[int]]:
    """Parse window index labels (e.g. '1-5', '6-10') into x positions and row order.

    Returns:
        x_positions: [0, t_end_1, t_end_2, ...] sorted by t_end.
        row_order: Corresponding row indices into market_share_windows.
    """
    parsed: List[Tuple[int, int]] = []
    for i, idx in enumerate(market_share_windows.index):
        parts = str(idx).strip().split("-")
        if len(parts) >= 2:
            t_end = int(parts[-1])
            parsed.append((t_end, i))
    parsed.sort(key=lambda p: p[0])
    x_positions = [0] + [p[0] for p in parsed]
    row_order = [p[1] for p in parsed]
    return x_positions, row_order


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def plot_market_share(
    market_share: pd.DataFrame,
    save_path: str | Path | None = None,
    title: str = "Market Share Progression",
    figsize: Tuple[int, int] = (12, 7),
    generator_names: Optional[Dict[str, str]] = None,
    generator_introduce_from: Optional[Dict[str, int]] = None,
    generator_colors: Optional[Dict[str, str]] = None,
):
    """Plot cumulative market share progression as a line chart.

    Every model starts at 0% at t=0. Simulation step 0 is shown as t=1, so
    the x-axis runs 0, 1, ..., T. If generator_introduce_from is set, each
    generator's line is only drawn from its introduce_from timestep onward.

    Args:
        market_share: DataFrame from :func:`~post_simulation.market_share.compute_market_share`.
        save_path: If provided, save the figure to this path.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        generator_names: Optional mapping from generator_id to display name.
        generator_introduce_from: Optional mapping generator_id -> introduce_from timestep.
        generator_colors: Optional mapping from generator_id to color.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    to_plot = _market_share_for_plot(market_share)
    if generator_introduce_from:
        to_plot = _mask_pre_introduction_display(to_plot, generator_introduce_from)
        for col in to_plot.columns:
            intro = generator_introduce_from.get(col, 0)
            if intro > 0 and (intro + 1) in to_plot.index:
                to_plot.loc[intro + 1, col] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    x = to_plot.index
    if hasattr(to_plot.index, "dtype") and to_plot.index.dtype.kind in "iu":
        x = to_plot.index.astype(int)

    for col in to_plot.columns:
        label = generator_names.get(col, col) if generator_names else col
        color = generator_colors.get(col) if generator_colors else None
        ax.plot(x, to_plot[col], marker="o", markersize=3, label=label, color=color)

    ax.set_xlabel("Simulation Time Step (t)", fontsize=12)
    ax.set_ylabel("Market Share (%)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0, 100)

    for intro_t in _unique_introduce_from_steps(generator_introduce_from):
        ax.axvline(x=intro_t + 1, linestyle="--", color="gray", alpha=0.7, zorder=0)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Market share plot saved to: {save_path}")

    plt.close(fig)
    return fig


def plot_market_share_windows(
    market_share_windows: pd.DataFrame,
    save_path: str | Path | None = None,
    title: str = "Market Share by Time Window",
    figsize: Tuple[int, int] = (12, 7),
    generator_names: Optional[Dict[str, str]] = None,
    generator_introduce_from: Optional[Dict[str, int]] = None,
    generator_colors: Optional[Dict[str, str]] = None,
):
    """Plot agent market share progression across time windows as a line chart.

    Every model starts at 0% at t=0. Each window's data point is placed at the
    right end of the window (e.g. window "1-5" at t=5). If
    generator_introduce_from is set, each generator's line is only drawn from
    its introduce_from timestep onward.

    Args:
        market_share_windows: DataFrame from
            :func:`~post_simulation.market_share.compute_market_share_windows`
            (index = window labels like "1-5", "6-10", columns = generator ids).
        save_path: If provided, save the figure to this path.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        generator_names: Optional mapping from generator_id to display name.
        generator_introduce_from: Optional mapping generator_id -> introduce_from timestep.
        generator_colors: Optional mapping from generator_id to color.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    x_positions, row_order = _window_labels_to_t_ends(market_share_windows)
    if len(x_positions) <= 1:
        row_order = []

    fig, ax = plt.subplots(figsize=figsize)

    for col in market_share_windows.columns:
        intro = generator_introduce_from.get(col, 0) if generator_introduce_from else 0
        y_values = [np.nan if intro > 0 else 0.0]
        for j, i in enumerate(row_order):
            t_end = x_positions[j + 1]
            val = float(market_share_windows.iloc[i][col])
            y_values.append(np.nan if t_end < intro else val)
        label = generator_names.get(col, col) if generator_names else col
        color = generator_colors.get(col) if generator_colors else None
        ax.plot(
            x_positions, y_values, marker="o", markersize=6, label=label, color=color
        )

    ax.set_xlabel("Simulation Time Step (t)", fontsize=12)
    ax.set_ylabel("Market Share (%)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)

    for intro_t in _unique_introduce_from_steps(generator_introduce_from):
        ax.axvline(x=intro_t, linestyle="--", color="gray", alpha=0.7, zorder=0)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Market share (windows) plot saved to: {save_path}")

    plt.close(fig)
    return fig


def plot_market_share_windows_stacked(
    market_share_windows: pd.DataFrame,
    save_path: str | Path | None = None,
    title: str = "Market Share by Time Window (Stacked)",
    figsize: Tuple[int, int] = (12, 7),
    generator_names: Optional[Dict[str, str]] = None,
    generator_introduce_from: Optional[Dict[str, int]] = None,
    generator_colors: Optional[Dict[str, str]] = None,
    generator_order: Optional[List[str]] = None,
    legend_fontsize: int = 8,
    legend_ncol: Optional[int] = None,
):
    """Plot a stacked area chart of market share across time windows.

    All models stack to 100% at each window. Generators are ordered in the
    stack by cumulative market share (highest at bottom). Models introduced
    after t=0 are excluded from the initial equal share at t=0.

    Args:
        market_share_windows: DataFrame from
            :func:`~post_simulation.market_share.compute_market_share_windows`
            (index = window labels like "1-5", "6-10", columns = generator ids).
        save_path: If provided, save the figure to this path.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        generator_names: Optional mapping from generator_id to display name.
        generator_introduce_from: Optional mapping generator_id -> introduce_from timestep.
        generator_colors: Optional mapping from generator_id to color.
        generator_order: Optional list of generator_ids to fix legend order.
        legend_fontsize: Font size for legend text.
        legend_ncol: Number of columns in the legend.

    Returns:
        matplotlib Figure, or None if there are no windows to plot.
    """
    import matplotlib.pyplot as plt

    x_positions, row_order = _window_labels_to_t_ends(market_share_windows)
    if len(x_positions) <= 1:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    num_windows = len(row_order)
    x_values = x_positions[1:]

    data_matrix = np.zeros((num_windows, len(market_share_windows.columns)))
    for j, i in enumerate(row_order):
        data_matrix[j, :] = market_share_windows.iloc[i].values

    row_sums = data_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    data_matrix = data_matrix / row_sums * 100.0

    cumulative_share = data_matrix.sum(axis=0)
    sorted_indices = np.argsort(-cumulative_share)
    sorted_columns = [market_share_windows.columns[i] for i in sorted_indices]
    data_matrix_sorted = data_matrix[:, sorted_indices]

    n_generators = len(sorted_columns)
    first_row = np.zeros((1, n_generators))

    if generator_introduce_from:
        initial_models = [
            col for col in sorted_columns if generator_introduce_from.get(col, 0) <= 0
        ]
        n_initial = len(initial_models)
        if n_initial > 0:
            equal_share = 100.0 / n_initial
            for i, col in enumerate(sorted_columns):
                if col in initial_models:
                    first_row[0, i] = equal_share
    else:
        first_row = np.full((1, n_generators), 100.0 / n_generators)

    x_values_with_zero = np.concatenate([[0], x_values])
    data_matrix_with_zero = np.vstack([first_row, data_matrix_sorted])

    labels = [
        generator_names.get(col, col) if generator_names else col
        for col in sorted_columns
    ]

    colors = None
    if generator_colors:
        colors = [generator_colors.get(col, None) for col in sorted_columns]
        if None in colors:
            colors = None

    ax.stackplot(
        x_values_with_zero,
        data_matrix_with_zero.T,
        labels=labels,
        colors=colors,
        alpha=0.8,
    )

    ax.set_xlabel("Simulation Time Step (t)", fontsize=12)
    ax.set_ylabel("Market Share (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0, right=x_values[-1] if x_values else 1)

    if generator_order:
        handles, legend_labels = ax.get_legend_handles_labels()
        gen_to_handle = {
            col: (handles[i], labels[i]) for i, col in enumerate(sorted_columns)
        }
        reordered_handles, reordered_labels = [], []
        for col in generator_order:
            if col in gen_to_handle:
                h, l = gen_to_handle[col]
                reordered_handles.append(h)
                reordered_labels.append(l)
        for col in sorted_columns:
            if col not in generator_order:
                h, l = gen_to_handle[col]
                reordered_handles.append(h)
                reordered_labels.append(l)
        ax.legend(
            reordered_handles,
            reordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=(
                legend_ncol
                if legend_ncol is not None
                else min(len(reordered_labels), 5)
            ),
            fontsize=legend_fontsize,
            frameon=False,
        )
    else:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=legend_ncol if legend_ncol is not None else min(len(labels), 5),
            fontsize=legend_fontsize,
            frameon=False,
        )

    # Mark late-introduced models with a vertical line at their introduction window start
    window_starts = []
    for idx in market_share_windows.index:
        parts = str(idx).strip().split("-")
        if len(parts) >= 2:
            window_starts.append(int(parts[0]) - 1)
        else:
            window_starts.append(0)

    for col_idx, col in enumerate(sorted_columns):
        original_idx = list(market_share_windows.columns).index(col)
        col_data = data_matrix[:, original_idx]
        intro_window_idx = None
        for i, val in enumerate(col_data):
            if val > 1.0 and (i == 0 or col_data[i - 1] < 1.0):
                intro_window_idx = i
                break
        if intro_window_idx is not None and intro_window_idx > 0:
            intro_t = window_starts[intro_window_idx]
            ax.axvline(
                x=intro_t,
                linestyle="--",
                color="gray",
                alpha=0.7,
                linewidth=2,
                zorder=10,
            )
            label = generator_names.get(col, col) if generator_names else col
            ax.text(
                intro_t - 2,
                2,
                label,
                rotation=90,
                verticalalignment="bottom",
                horizontalalignment="right",
                fontsize=9,
                fontweight="bold",
                color="black",
                alpha=1.0,
            )

    if generator_introduce_from:
        for intro_t in _unique_introduce_from_steps(generator_introduce_from):
            if intro_t > 0:
                ax.axvline(
                    x=intro_t,
                    linestyle="--",
                    color="gray",
                    alpha=0.5,
                    linewidth=0.8,
                    zorder=9,
                )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Market share (windows, stacked) plot saved to: {save_path}")

    plt.close(fig)
    return fig
