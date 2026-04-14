r"""
Post-simulation customer retention rate (CRR) metrics.

Let $\tau_{u,a} = \min \{\, t \in \{1,\dots,T\} : q_u(t)=a \,\}$ denote the first timestep at which user $u$ selects agent $a$, if such a timestep exists.
Given a window length $m$, the \emph{user-level retention} of agent $a$ for user $u$ is
\begin{equation}
    \text{CR}_{u,a}(m) =
    \frac{1}{m}\sum_{t=1}^{m}
    \mathbf{1}\!\left[q_u(\tau_{u,a}+t)=a\right],
\label{eq:retention-user}
\end{equation}
which measures the fraction of the next $m$ interactions, following first adoption, in which $u$ continues selecting $a$.
%
Aggregating across users yields the \emph{agent-level retention}
\begin{equation}
    \text{CR}_{a}(m) =
    \frac{1}{|U_a|}
    \sum_{u\in U_a}\text{CR}_{u,a}(m),
\label{eq:retention-agent}
\end{equation}
where $U_a := \{ u \in U : \tau_{u,a}\ \text{is defined} \}$ is the set of users who have tried agent $a$ at least once.
A higher $\text{CR}_{a}(m)$ indicates that users who initially sample $a$ tend to continue allocating a larger fraction of their subsequent interactions to that agent.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_crr_windowed(
    log_path: str,
    t_start: int,
    t_end: int,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Compute agent-level customer retention rate CR_a(m) within a time window.

    For each (user, agent) pair within [t_start, t_end]:
    1. Find tau_{u,a} = first timestep in the window where user u selects agent a.
    2. Collect all of user u's subsequent interactions (after tau_{u,a}) within
       the window.
    3. CR_{u,a} = fraction of those subsequent interactions where u selected a.
    4. CR_a = mean of CR_{u,a} across all users who adopted a AND had at least
       one subsequent interaction in the window (so retention is measurable).

    Args:
        log_path: Path to simulation_log.csv.
        t_start: Start of the window (inclusive), 0-indexed.
        t_end: End of the window (inclusive), 0-indexed.

    Returns:
        Tuple of:
        - agent_crr: Dict mapping generator_id -> CR_a (agent-level retention).
        - user_crr: Dict mapping generator_id -> list of per-user CR_{u,a} values.
    """
    df = pd.read_csv(log_path, usecols=["t", "user_id", "generator_id"])
    df = df[(df["t"] >= t_start) & (df["t"] <= t_end)].copy()
    df = df.sort_values(["user_id", "t"]).reset_index(drop=True)

    all_generators = sorted(df["generator_id"].unique())
    all_users = sorted(df["user_id"].unique())

    user_crr: Dict[str, List[float]] = {g: [] for g in all_generators}
    agent_crr: Dict[str, float] = {}

    for agent in all_generators:
        per_user_values = []
        for user in all_users:
            user_interactions = df[df["user_id"] == user].reset_index(drop=True)

            first_adoption = user_interactions[
                user_interactions["generator_id"] == agent
            ]
            if first_adoption.empty:
                continue

            tau_idx = first_adoption.index[0]
            subsequent = user_interactions.iloc[tau_idx + 1 :]
            if subsequent.empty:
                continue

            m = len(subsequent)
            retained = (subsequent["generator_id"] == agent).sum()
            cr_u_a = retained / m

            per_user_values.append(cr_u_a)

        user_crr[agent] = per_user_values

        if per_user_values:
            agent_crr[agent] = np.mean(per_user_values)
        else:
            agent_crr[agent] = float("nan")

    return agent_crr, user_crr


def compute_crr_report(
    log_path: str,
    windows: List[Tuple[int, int]],
    generator_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Compute CR_a for multiple windows and return a summary DataFrame.

    Args:
        log_path: Path to simulation_log.csv.
        windows: List of (t_start, t_end) tuples (0-indexed, inclusive).
        generator_names: Optional mapping from generator_id to display name.

    Returns:
        DataFrame with rows = windows, columns = generators, values = CR_a.
    """
    rows = []
    for t_start, t_end in windows:
        agent_crr, _ = compute_crr_windowed(log_path, t_start, t_end)
        agent_crr["window"] = f"{t_start + 1}-{t_end + 1}"
        rows.append(agent_crr)

    result = pd.DataFrame(rows).set_index("window")
    gen_cols = sorted([c for c in result.columns])
    result = result[gen_cols]

    if generator_names:
        result = result.rename(columns=generator_names)

    return result
