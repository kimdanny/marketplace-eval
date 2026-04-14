"""
Minimal default demo — no customization beyond the required judge definition.

All framework defaults are used:
* UserProfile        — built-in preference scoring and generator selection
* UserPopulation     — weighted random sampling

The only required user-supplied code is a concrete LLMJudgeAgent subclass,
since LLMJudgeAgent is abstract and must be registered before loading config.

Usage:
    python examples/demo_default.py
    python examples/demo_default.py --config configs/default.yaml --output-dir results/default
"""

import argparse
from pathlib import Path

import pandas as pd

from marketplace_eval.agents.llm_judge_agent import LLMJudgeAgent, register_judge
from marketplace_eval.post_simulation.market_share import compute_market_share
from marketplace_eval.system.system import System
from marketplace_eval.system.types import GenerationResult, JudgeFeedback, UserQuery


@register_judge("default")
class DefaultJudge(LLMJudgeAgent):
    """Rates answers 0–10 and normalises to [-1, 1]."""

    PROMPT = (
        "Question: {question}\n"
        "Ground-truth: {ground_truth}\n"
        "Answer: {answer}\n\n"
        "Rate the answer from 0 (wrong) to 10 (perfect). Reply with a single integer."
    )

    def format_prompt(
        self, *, generation: GenerationResult, user_query: UserQuery | None = None
    ) -> str:
        return self.PROMPT.format(
            question=generation.question,
            ground_truth=(user_query.answer if user_query else None)
            or generation.metadata.get("ground_truth_answer", ""),
            answer=generation.answer,
        )

    def parse_llm_response(self, response: str) -> float:
        for token in response.split():
            try:
                v = int(token)
                if 0 <= v <= 10:
                    return v / 5.0 - 1.0  # maps 0→-1, 5→0, 10→1
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
        return JudgeFeedback(
            score=score,
            rationale=raw_response.strip()[:100],
            retriever_scores={},
            generator_feedback={},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "configs" / "default.yaml"),
    )
    parser.add_argument("--output-dir", default="results/default")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "simulation_log.csv"

    system = System()
    system.initialize_from_config(args.config)
    system.initialize()
    system.run(log_csv_path=log_path)

    # --- Post-simulation analysis ---
    df = pd.read_csv(log_path)

    # Final cumulative market share (%)
    market_share = compute_market_share(log_path)
    final_share = market_share.iloc[-1].sort_values(ascending=False)

    # Mean judge score per generator (score is in [-1, 1])
    mean_scores = (
        df.groupby("generator_id")["score"].mean().sort_values(ascending=False)
    )

    print("\n=== Market Share (final) ===")
    for gen_id, share in final_share.items():
        print(f"  {gen_id}: {share:.1f}%")

    print("\n=== Mean Judge Score ===")
    for gen_id, score in mean_scores.items():
        print(f"  {gen_id}: {score:.3f}")


if __name__ == "__main__":
    main()
