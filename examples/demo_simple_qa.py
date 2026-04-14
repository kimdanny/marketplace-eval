"""
Demo script for running the Simple QA market simulation.

This script demonstrates how a framework user defines a custom judge agent
by subclassing ``LLMJudgeAgent`` and registering it before running the
simulation.  The ``SimpleQAJudgeAgent`` defined below is specific to the
SimpleQA factuality benchmark and is **not** part of the core framework.

Usage:
    python demo_simple_qa.py
    python demo_simple_qa.py --config configs/simple_qa_simulation.yaml
    python demo_simple_qa.py --config configs/simple_qa_simulation.yaml --output-dir results/simple_qa
    python demo_simple_qa.py --config configs/simple_qa_simulation.yaml --output-dir results/simple_qa --window-interval 100
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from marketplace_eval.agents.generator_agent import GeneratorAgent
from marketplace_eval.agents.llm_judge_agent import LLMJudgeAgent, register_judge
from marketplace_eval.prompts.simple_qa_judge_prompt import GRADER_TEMPLATE
from marketplace_eval.system.types import GenerationResult, JudgeFeedback, UserQuery
from marketplace_eval.system.system import System
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
# Custom judge: SimpleQA grading rubric (user-defined, not part of the core
# framework).  Any user can follow this pattern to create their own judge.
# ---------------------------------------------------------------------------

_GRADE_MAP = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def _extract_grade(response: str) -> str:
    """Extract grade (A, B, or C) from LLM judge response."""
    response = response.strip().upper()
    if len(response) == 1 and response in ("A", "B", "C"):
        return response
    for char in response:
        if char in ("A", "B", "C"):
            return char
    return "C"


def _grade_to_score(grade: str) -> float:
    """Convert a grade letter to a numeric score. A=1.0, B/C=-1.0."""
    return 1.0 if grade == "A" else -1.0


@register_judge("simple_qa")
class SimpleQAJudgeAgent(LLMJudgeAgent):
    """Judge agent using the SimpleQA grading rubric.

    Grades LLM answers as CORRECT (A), INCORRECT (B), or NOT_ATTEMPTED (C)
    against a ground-truth target, following the protocol from:

        SimpleQA: Measuring short-form factuality in large language models
        (Wei et al., 2024)
    """

    def __init__(self, node_id: str, **kwargs: Any):
        super().__init__(node_id, **kwargs)

    def format_prompt(
        self,
        *,
        generation: GenerationResult,
        user_query: UserQuery | None = None,
    ) -> str:
        ground_truth = None
        if user_query is not None:
            ground_truth = user_query.answer
        if ground_truth is None:
            ground_truth = generation.metadata.get("ground_truth_answer", "")

        return GRADER_TEMPLATE.format(
            question=generation.question,
            target=ground_truth,
            predicted_answer=generation.answer,
        )

    def parse_llm_response(self, response: str) -> float:
        return _grade_to_score(_extract_grade(response))

    def build_judge_feedback(
        self,
        *,
        score: float,
        raw_response: str,
        generation: GenerationResult,
        user_query: UserQuery | None = None,
    ) -> JudgeFeedback:
        grade = _extract_grade(raw_response)
        rationale = (
            f"Grade: {grade} ({_GRADE_MAP.get(grade, grade)}) "
            f"(raw response: {raw_response.strip()[:100]})"
        )

        # If your system is RAG, you can configure retriever evaluation as well as generator output evaluation.
        retriever_scores: dict[str, float] = {}
        if generation.retrievals:
            for call in generation.retrievals:
                r_score = sum(call.scores) / len(call.scores) if call.scores else 0.0
                retriever_scores[call.retriever_id] = r_score

        return JudgeFeedback(
            score=score,
            rationale=rationale,
            retriever_scores=retriever_scores,
            generator_feedback={"grade": grade, "raw_response": raw_response.strip()},
        )


# ---------------------------------------------------------------------------
# SimpleQA experiment plot configuration
#
# These settings reproduce the figures in the SIGIR 2026 paper.  Customize
# generator_names, generator_colors, and generator_order to match your own
# experiment setup when running with a different set of models.
# ---------------------------------------------------------------------------

SIMPLE_QA_GENERATOR_NAMES: Dict[str, str] = {
    "generator_gemini_2_5_flash_lite": "Gemini 2.5 Flash Lite",
    "generator_qwen3_235b_a22b_2507": "Qwen3 235B A22B Instruct 2507",
    "generator_kimi_k2_5": "Kimi K2.5",
    "generator_gpt_oss_120b": "GPT-OSS-120B",
    "generator_deepseek_v3_2": "DeepSeek V3.2",
    "generator_llama_3_3_70b_instruct": "Llama 3.3 70B Instruct",
    "generator_grok_4_1_fast": "Grok 4.1 Fast",
}

SIMPLE_QA_GENERATOR_COLORS: Dict[str, str] = {
    "generator_deepseek_v3_2": "#E74C3C",
    "generator_qwen3_235b_a22b_2507": "#795548",
    "generator_gemini_2_5_flash_lite": "#7F8C8D",
    "generator_kimi_k2_5": "#3F51B5",
    "generator_gpt_oss_120b": "#BDC3C7",
    "generator_llama_3_3_70b_instruct": "#F1C40F",
    "generator_grok_4_1_fast": "#009688",
}

# Determines legend order in stacked plots (does not affect stacking order).
SIMPLE_QA_GENERATOR_ORDER: List[str] = [
    "generator_deepseek_v3_2",
    "generator_qwen3_235b_a22b_2507",
    "generator_kimi_k2_5",
    "generator_llama_3_3_70b_instruct",
    "generator_gemini_2_5_flash_lite",
    "generator_gpt_oss_120b",
    "generator_grok_4_1_fast",
]


def plot_publication_figures(
    market_share_windows,
    output_dir: Path,
    generator_introduce_from: Optional[Dict[str, int]],
) -> None:
    """Save publication-quality stacked market share PDF for SimpleQA experiments.

    Uses the figure layout from the SIGIR 2026 paper: wide aspect ratio
    (12×4 inches), 4-column legend below the axes, and model-specific colors.
    Output is saved as both PDF (vector) and PNG (raster) to output_dir.

    Args:
        market_share_windows: DataFrame from compute_market_share_windows.
        output_dir: Directory to write figures into.
        generator_introduce_from: Mapping generator_id -> introduce_from timestep,
            used to annotate the late-entry vertical line.
    """
    import matplotlib

    # Apply publication-quality rcParams for this block only
    with matplotlib.rc_context(
        {
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.labelsize": 20,
        }
    ):
        for ext in ("pdf", "png"):
            plot_market_share_windows_stacked(
                market_share_windows,
                save_path=output_dir / f"market_share_windows_stacked_pub.{ext}",
                generator_names=SIMPLE_QA_GENERATOR_NAMES,
                generator_introduce_from=generator_introduce_from,
                generator_colors=SIMPLE_QA_GENERATOR_COLORS,
                generator_order=SIMPLE_QA_GENERATOR_ORDER,
                legend_fontsize=12,
                legend_ncol=4,
                figsize=(12, 4),
            )


# ---------------------------------------------------------------------------
# Main simulation driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run market simulation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path(__file__).parent.parent / "configs" / "simple_qa_simulation.yaml"
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
        help=(
            "Window size for windowed market share (e.g. 10 -> windows 1-10, 11-20, ...). "
            "The SIGIR 2026 experiments used --window-interval 10 for a 200-step simulation."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = output_dir / "simulation_log.csv"

    # Initialize system
    print("=" * 60)
    print("Initializing system from config...")
    print("=" * 60)
    system = System()
    system.initialize_from_config(args.config)
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

        # Standard plots
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

        # Publication-quality figures using SimpleQA experiment configuration
        plot_publication_figures(
            market_share_windows, output_dir, generator_introduce_from
        )


if __name__ == "__main__":
    main()
