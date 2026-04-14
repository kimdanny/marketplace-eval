"""
Demo script for running the RAG market simulation with FAISS-backed retrievers.

This demo also shows how framework users can **override** default behaviours
by subclassing core classes:

* ``UserProfile``       — override ``choose_generator`` and ``update_preference``
* ``UserPopulation``    — override ``sample_profiles``
* ``BaseRetrievalPlanner`` — register a custom retrieval strategy with ``@register_planner``
* ``RouterAgent``       — register a custom router strategy with ``@register_router``

Usage:
    python demo_sample.py
    python demo_sample.py --config configs/sample_simulation.yaml
    python demo_sample.py --config configs/sample_simulation.yaml --output-dir results/sample
"""

import argparse
import asyncio
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from marketplace_eval.agents.generator_agent import (
    BaseRetrievalPlanner,
    GeneratorAgent,
    register_planner,
)
from marketplace_eval.agents.llm_judge_agent import LLMJudgeAgent, register_judge
from marketplace_eval.agents.router_agent import RouterAgent, register_router
from marketplace_eval.system.types import (
    GenerationResult,
    JudgeFeedback,
    UserProfile,
    UserQuery,
)
from marketplace_eval.system.system import System
from marketplace_eval.system.user_population import UserPopulation
from marketplace_eval.post_simulation.market_share import (
    compute_market_share,
    compute_market_share_windows,
    get_windows_from_interval,
)
from marketplace_eval.post_simulation.plot import (
    plot_market_share,
    plot_market_share_windows,
    plot_market_share_windows_stacked,
)


# ---------------------------------------------------------------------------
# Custom UserProfile — override choose_generator and update_preference.
# Framework users can subclass UserProfile and inject their own logic.
# ---------------------------------------------------------------------------


@dataclass
class SoftmaxUserProfile(UserProfile):
    """A user profile that uses softmax-based selection and momentum-based updates.

    * ``choose_generator`` — selects a generator by sampling from a softmax
      distribution over preference scores (temperature-controlled).
    * ``update_preference`` — applies an exponential moving average (EMA)
      update instead of the default linear drift.
    """

    temperature: float = 0.5
    ema_alpha: float = 0.3

    def choose_generator(
        self,
        rng: random.Random,
        epsilon: float = 0.2,
        available_ids: Optional[List[str]] = None,
    ) -> str:
        """Softmax sampling over preference scores with temperature scaling."""
        generators = list(self.preference_scores.keys())
        if available_ids is not None:
            generators = [g for g in generators if g in available_ids]
            if not generators:
                generators = list(available_ids)
        if not generators:
            raise ValueError("No generators available for selection")

        scores = [self.preference_scores.get(g, 0.0) for g in generators]
        max_score = max(scores)
        exp_scores = [
            math.exp((s - max_score) / max(self.temperature, 1e-8)) for s in scores
        ]
        total = sum(exp_scores)
        probabilities = [e / total for e in exp_scores]

        return rng.choices(generators, weights=probabilities, k=1)[0]

    def update_preference(
        self, generator_id: str, score: float, drift_rate: float = 0.1
    ):
        """EMA update: new_pref = alpha * normalised_score + (1 - alpha) * old_pref."""
        normalised_score = (score + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
        old = self.preference_scores.get(generator_id, 0.01)
        self.preference_scores[generator_id] = (
            self.ema_alpha * normalised_score + (1 - self.ema_alpha) * old
        )
        total = sum(self.preference_scores.values())
        if total > 0:
            for key in list(self.preference_scores.keys()):
                self.preference_scores[key] /= total


def _build_custom_profiles(
    generator_ids: List[str], total_num_users: int
) -> List[SoftmaxUserProfile]:
    """Create SoftmaxUserProfile instances with uniform initial preferences."""
    equal_prob = 1.0 / len(generator_ids) if generator_ids else 1.0
    initial_preferences = {gid: equal_prob for gid in generator_ids}
    sampling_prob = 1.0 / total_num_users

    return [
        SoftmaxUserProfile(
            user_id=f"user_{i + 1}",
            preference_scores=initial_preferences.copy(),
            sampling_probability=sampling_prob,
            rng_state={},
            temperature=0.5,
            ema_alpha=0.3,
        )
        for i in range(total_num_users)
    ]


# ---------------------------------------------------------------------------
# Custom UserPopulation — override sample_profiles.
# Framework users can subclass UserPopulation to change how users are sampled.
# ---------------------------------------------------------------------------


class RoundRobinPopulation(UserPopulation):
    """A population that cycles through profiles in round-robin order.

    Instead of weighted random sampling, each call to ``sample_profiles``
    yields the next *n* profiles from a deterministic rotation.  This
    guarantees every profile is exercised equally over the course of the
    simulation — useful for fairness-sensitive evaluations.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._rr_index = 0

    def sample_profiles(self, n: int) -> List[UserProfile]:
        """Return the next *n* profiles by cycling through the roster."""
        sampled: List[UserProfile] = []
        for _ in range(n):
            sampled.append(self.profiles[self._rr_index % len(self.profiles)])
            self._rr_index += 1
        return sampled


# ---------------------------------------------------------------------------
# Custom RetrievalPlanner — register with @register_planner before the system
# loads the config, then reference via retrieval_strategy.type in YAML.
# ---------------------------------------------------------------------------


@register_planner("random_subset")
class RandomSubsetPlanner(BaseRetrievalPlanner):
    """Queries a random subset of available routers and retrievers.

    Unlike ``NaiveRetrievalPlanner`` (which always queries *all* sources),
    this planner caps the number of sources contacted per step.  This is
    useful for simulating budget-constrained retrieval or studying the
    effect of partial information on answer quality.

    Config keys (all optional):

    * ``max_sources`` — maximum number of routers/retrievers to query
      per step (default: ``1``).
    * ``router_top_k`` — forwarded to each router as the number of
      retrievers to select (default: ``1``).

    YAML usage::

        retrieval_strategy:
          type: random_subset
          max_sources: 2
          router_top_k: 1
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.max_sources: int = max(1, int(self.config.get("max_sources", 1)))
        self.router_top_k: int = int(self.config.get("router_top_k", 1))

    async def run(
        self,
        generator: GeneratorAgent,
        question: str,
        system: Any,
    ) -> List[Any]:
        from marketplace_eval.agents.router_agent import RouterAgent
        from marketplace_eval.system.types import RetrievalCall

        combined = [*generator._connected_routers(), *generator._connected_retrievers()]
        random.shuffle(combined)
        selected = combined[: self.max_sources]

        tasks = []
        for node in selected:
            if isinstance(node, RouterAgent):
                tasks.append(
                    generator._invoke_router(
                        node, question, system, top_k=self.router_top_k
                    )
                )
            else:
                tasks.append(
                    generator._invoke_retriever(node, question, router_id=None)
                )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks)
        retrieval_calls: List[Any] = []
        for result in results:
            if isinstance(result, list):
                retrieval_calls.extend(result)
            elif isinstance(result, RetrievalCall):
                retrieval_calls.append(result)
        return retrieval_calls


# ---------------------------------------------------------------------------
# Custom RouterAgent — register with @register_router before the system
# loads the config, then reference via router_strategy in YAML.
# ---------------------------------------------------------------------------


@register_router("ucb")
class UCBRouterAgent(RouterAgent):
    """Upper Confidence Bound (UCB1) router strategy.

    Selects retrievers using the UCB1 formula::

        score(r) = avg_reward(r) + C * sqrt(ln(N + 1) / (n_r + 1))

    where N is the total number of router invocations, n_r is the number of
    times retriever *r* has been selected, and C is the exploration weight.

    Compared to the default epsilon-greedy router, UCB provides a principled
    exploration-exploitation trade-off: retrievers with high uncertainty
    (low n_r) receive an exploration bonus that diminishes naturally as they
    are tried more often — no fixed exploration probability is needed.

    Config keys (all optional):

    * ``exploration_weight`` — the C constant controlling exploration strength
      (default: ``1.0``).

    YAML usage::

        id: router_main
        type: router
        params:
          router_strategy: ucb
          exploration_weight: 1.0
    """

    def __init__(self, node_id: str, *, exploration_weight: float = 1.0, **kwargs):
        super().__init__(node_id, **kwargs)
        self.exploration_weight = exploration_weight
        self._total_invocations: int = 0

    async def invoke(
        self,
        *,
        query: str,
        generator_id: str,
        top_k: int | None = 1,
        system: Any = None,
    ) -> List[str]:
        retrievers = self._available_retrievers()
        if not retrievers:
            return []

        effective_top_k: int
        if top_k is None or top_k <= 0:
            effective_top_k = len(retrievers)
        else:
            effective_top_k = min(top_k, len(retrievers))

        self._total_invocations += 1

        def ucb_score(rid: str) -> float:
            n = self.retriever_counts.get(rid, 0)
            avg = self.retriever_scores.get(rid, 0.0) / n if n > 0 else 0.0
            exploration_bonus = self.exploration_weight * math.sqrt(
                math.log(self._total_invocations + 1) / (n + 1)
            )
            return avg + exploration_bonus

        ranked = sorted(retrievers, key=ucb_score, reverse=True)
        selected = ranked[:effective_top_k]

        for retriever_id in selected:
            self.retriever_counts[retriever_id] = (
                self.retriever_counts.get(retriever_id, 0) + 1
            )
        return selected


# ---------------------------------------------------------------------------
# Custom judge — register before the system loads the config.
# This is the minimal code a framework user needs to write.
# ---------------------------------------------------------------------------

SAMPLE_JUDGE_PROMPT = """\
You are an evaluation judge. Given a question, a ground-truth answer, and a \
predicted answer, rate the predicted answer on a scale of 0 to 10 where:
  10 = perfectly correct and complete
   0 = completely wrong or irrelevant

Question: {question}
Ground-truth answer: {ground_truth}
Predicted answer: {predicted_answer}

Respond with ONLY a single integer between 0 and 10.
""".strip()


@register_judge("sample")
class SampleJudgeAgent(LLMJudgeAgent):
    """A simple 0-10 scoring judge for demonstration purposes."""

    def format_prompt(
        self, *, generation: GenerationResult, user_query: UserQuery | None = None
    ) -> str:
        ground_truth = (
            user_query.answer if user_query else None
        ) or generation.metadata.get("ground_truth_answer", "")
        return SAMPLE_JUDGE_PROMPT.format(
            question=generation.question,
            ground_truth=ground_truth,
            predicted_answer=generation.answer,
        )

    def parse_llm_response(self, response: str) -> float:
        """Extract the first integer (0-10) from the response, normalise to [-1, 1]."""
        for token in response.strip().split():
            try:
                value = int(token)
                if 0 <= value <= 10:
                    return value / 5.0 - 1.0  # 0→-1.0, 5→0.0, 10→1.0
            except ValueError:
                continue
        return -1.0

    def build_judge_feedback(
        self,
        *,
        score: float,
        raw_response: str,
        generation: GenerationResult,
        user_query: UserQuery | None = None,
    ) -> JudgeFeedback:
        retriever_scores: dict[str, float] = {}
        for call in generation.retrievals:
            retriever_scores[call.retriever_id] = (
                sum(call.scores) / len(call.scores) if call.scores else 0.0
            )

        return JudgeFeedback(
            score=score,
            rationale=f"LLM judge raw: {raw_response.strip()[:100]}",
            retriever_scores=retriever_scores,
            generator_feedback={"raw_response": raw_response.strip()},
        )


def main():
    parser = argparse.ArgumentParser(description="Run RAG market simulation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path(__file__).parent.parent / "configs" / "sample_simulation.yaml"
        ),
        help="Path to simulation config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--window-interval",
        type=int,
        default=None,
        metavar="N",
        help="Window size for market share progression (e.g. 5 -> windows 1-5, 6-10, ...)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = output_dir / "simulation_log.csv"

    # Initialize system (retriever FAISS indices are built during config loading)
    print("=" * 60)
    print("Initializing system from config...")
    print("=" * 60)
    system = System()
    system.initialize_from_config(args.config)

    # Customizing UserProfile and UserPopulation classes.
    # SoftmaxUserProfile  -> overrides choose_generator & update_preference
    # RoundRobinPopulation -> overrides sample_profiles
    initial_generator_ids = [
        gid
        for gid in system.generator_ids
        if system.generator_introduce_from.get(gid, 0) == 0
    ]
    custom_profiles = _build_custom_profiles(
        initial_generator_ids, system.total_num_users
    )
    system.population = RoundRobinPopulation(
        profiles=custom_profiles,
        user_data=system.population.user_data,
        rng_seed=system.config.get("simulation", {}).get("seed"),
    )
    print(
        "\n** Using custom SoftmaxUserProfile (overridden choose_generator & update_preference) **"
    )
    print("** Using custom RoundRobinPopulation (overridden sample_profiles) **")
    print(
        "** RandomSubsetPlanner registered — use retrieval_strategy.type: random_subset in YAML **"
    )
    print("** UCBRouterAgent registered    — use router_strategy: ucb in YAML **")

    system.initialize()

    print("\nSystem State:")
    state = system.get_state()
    for key, value in state.items():
        print(f"  {key}: {value}")

    print(f"\nNumber of user profiles: {len(system.population.profiles)}")
    print(f"Number of questions in dataset: {len(system.population.user_data)}")

    # Run simulation with incremental CSV logging
    print("\n" + "=" * 60)
    print("Running simulation...")
    print("=" * 60)
    system.run(log_csv_path=log_csv_path)

    print(f"\nSimulation complete. Log saved to: {log_csv_path}")
    print(f"Total records: {len(system.logger.step_records)}")

    # Extract generator names from config for nice plot labels
    generator_names = {}
    for agent_id, agent in system.agents.items():
        if isinstance(agent, GeneratorAgent):
            generator_names[agent_id] = agent.name

    # Post-simulation analysis
    print("\n" + "=" * 60)
    print("Post-simulation analysis...")
    print("=" * 60)

    generator_introduce_from = system.generator_introduce_from or None
    gen_names = generator_names if generator_names else None

    # Cumulative market share over all timesteps
    market_share = compute_market_share(log_csv_path)
    market_share.to_csv(output_dir / "market_share.csv")
    plot_market_share(
        market_share,
        save_path=output_dir / "market_share.png",
        generator_names=gen_names,
        generator_introduce_from=generator_introduce_from,
    )

    # Windowed market share (only when --window-interval is provided)
    if args.window_interval:
        windows = get_windows_from_interval(log_csv_path, args.window_interval)
        market_share_windows = compute_market_share_windows(log_csv_path, windows)
        market_share_windows.to_csv(output_dir / "market_share_windows.csv")
        plot_market_share_windows(
            market_share_windows,
            save_path=output_dir / "market_share_windows.png",
            generator_names=gen_names,
            generator_introduce_from=generator_introduce_from,
        )
        plot_market_share_windows_stacked(
            market_share_windows,
            save_path=output_dir / "market_share_windows_stacked.png",
            generator_names=gen_names,
            generator_introduce_from=generator_introduce_from,
        )


if __name__ == "__main__":
    main()
