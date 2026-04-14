"""
Post-simulation market share metrics.

Market share for agent *a* in a window [t_start, t_end] is defined as:

    MS(a) = sum_{t=t_start}^{t_end} sum_{u in U} 1[q_u(t) = a]
            / sum_{t=t_start}^{t_end} sum_{u in U} 1

where q_u(t) denotes the agent selected by user u at timestep t.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def load_generator_introduce_from(config_path: str | Path) -> Dict[str, int]:
    """Load generator_id -> introduce_from timestep from a simulation config file.

    Reads graph.nodes; generators without 'introduce_from' default to 0.

    Args:
        config_path: Path to a YAML or JSON simulation config file.

    Returns:
        Dict mapping generator_id to its introduce_from timestep.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = f.read()
    if yaml is not None:
        config = yaml.safe_load(raw)
    else:
        config = json.loads(raw)
    out: Dict[str, int] = {}
    for node in config.get("graph", {}).get("nodes", []):
        if node.get("type", "").lower() == "generator":
            out[node["id"]] = node.get("introduce_from", 0)
    return out


def _load_log(log_path: str | Path) -> pd.DataFrame:
    """Load and validate simulation log CSV."""
    df = pd.read_csv(log_path)
    if "generator_id" not in df.columns or "t" not in df.columns:
        raise ValueError("Log CSV must contain 'generator_id' and 't' columns.")
    return df


def _market_share_in_window(
    df: pd.DataFrame,
    t_start: int,
    t_end: int,
    all_generators: List[str],
) -> pd.Series:
    """Compute market share over [t_start, t_end] (inclusive) from a log DataFrame."""
    mask = (df["t"] >= t_start) & (df["t"] <= t_end)
    window_df = df.loc[mask]
    if window_df.empty:
        return pd.Series({g: 0.0 for g in all_generators})
    total = len(window_df)
    counts = (
        window_df.groupby("generator_id").size().reindex(all_generators, fill_value=0)
    )
    return (counts / total * 100.0).fillna(0.0)


def compute_market_share(
    log_path: str | Path,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> pd.DataFrame:
    """Compute market share, optionally over a time window.

    Args:
        log_path: Path to the simulation log CSV.
        t_start: Start timestep (inclusive). If None, use first step in log.
        t_end: End timestep (inclusive). If None, use last step in log.

    Returns:
        - If both t_start and t_end are None: DataFrame indexed by ``t`` with one
          column per generator, cumulative market share at each time step (0-100).
        - If both t_start and t_end are set: DataFrame with a single row (window
          label as index) and one column per generator, market share in that
          window (0-100).
    """
    df = _load_log(log_path)
    all_generators = sorted(df["generator_id"].unique())

    if t_start is not None and t_end is not None:
        mask = (df["t"] >= t_start) & (df["t"] <= t_end)
        window_df = df.loc[mask]
        if window_df.empty:
            total = 0
            counts = {g: 0 for g in all_generators}
        else:
            total = len(window_df)
            counts = (
                window_df.groupby("generator_id")
                .size()
                .reindex(all_generators, fill_value=0)
            )
        share = counts / total * 100.0 if total else counts * 0.0
        index_name = f"{t_start}-{t_end}"
        return pd.DataFrame(
            [share.values],
            index=[index_name],
            columns=share.index,
        )

    if t_start is not None:
        df = df.loc[df["t"] >= t_start]
    if t_end is not None:
        df = df.loc[df["t"] <= t_end]
    if df.empty:
        return pd.DataFrame(columns=all_generators)

    all_steps = sorted(df["t"].unique())
    counts = (
        df.groupby(["t", "generator_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_steps, columns=all_generators, fill_value=0)
    )
    cum_counts = counts.cumsum()
    cum_total = cum_counts.sum(axis=1)
    market_share = cum_counts.div(cum_total, axis=0) * 100.0
    market_share.index.name = "t"
    return market_share


def get_windows_from_interval(
    log_path: str | Path,
    interval: int,
) -> List[Tuple[int, int]]:
    """Build non-overlapping windows of a fixed length from the log's time range.

    E.g. interval=5 with steps 0..19 gives (0,4), (5,9), (10,14), (15,19).
    Display labels are 1-based ("1-5", "6-10", ...) when used with
    :func:`compute_market_share_windows`.

    Args:
        log_path: Path to the simulation log CSV.
        interval: Number of steps per window.

    Returns:
        List of (t_start, t_end) in 0-indexed inclusive form.
    """
    if interval < 1:
        raise ValueError("window_interval must be at least 1")
    df = _load_log(log_path)
    max_t = int(df["t"].max())
    windows = []
    start = 0
    while start <= max_t:
        end = min(start + interval - 1, max_t)
        windows.append((start, end))
        start += interval
    return windows


def compute_market_share_windows(
    log_path: str | Path,
    windows: Sequence[Tuple[int, int]],
) -> pd.DataFrame:
    """Compute market share for multiple time windows.

    Args:
        log_path: Path to the simulation log CSV.
        windows: List of (t_start, t_end) 0-indexed inclusive ranges.

    Returns:
        DataFrame with index = window labels ("1-5", "6-10", ...), columns = generator ids,
        values = market share percentage (0-100) in that window.
    """
    df = _load_log(log_path)
    all_generators = sorted(df["generator_id"].unique())
    rows = []
    index_labels = []
    for t_start, t_end in windows:
        row = _market_share_in_window(df, t_start, t_end, all_generators)
        rows.append(row)
        index_labels.append(f"{t_start + 1}-{t_end + 1}")
    if not rows:
        return pd.DataFrame(columns=all_generators)
    out = pd.DataFrame(rows, index=index_labels)
    out.index.name = "window"
    return out


def parse_windows(spec: str) -> List[Tuple[int, int]]:
    """Parse a windows spec string into a list of 0-indexed (t_start, t_end) tuples.

    Accepts 1-based display form and converts to 0-indexed for computation:
        "1-5,6-10" -> [(0, 4), (5, 9)]

    Args:
        spec: Comma-separated window ranges in 1-based display form (e.g. "1-5,6-10").

    Returns:
        List of 0-indexed (t_start, t_end) tuples.
    """
    windows = []
    for part in spec.split(","):
        part = part.strip()
        m = re.match(r"(\d+)\s*-\s*(\d+)", part)
        if not m:
            raise ValueError(f"Invalid window spec: {part!r}. Use e.g. 1-5,6-10")
        a, b = int(m.group(1)), int(m.group(2))
        windows.append((a - 1, b - 1))  # 1-based -> 0-indexed
    return windows
