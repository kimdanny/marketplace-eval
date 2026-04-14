import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

import yaml

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from marketplace_eval.utils.llm_client import create_llm_client


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_config(config: dict, model_id: str) -> dict:
    """Get configuration for a specific model."""
    models = config.get("models", [])
    for model in models:
        if model["id"] == model_id:
            return model
    raise ValueError(f"Model '{model_id}' not found in config")


def load_dataset(dataset_path: str) -> list[dict]:
    """Load the Simple QA dataset."""
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(
                {"qid": row["qid"], "problem": row["problem"], "answer": row["answer"]}
            )
    return dataset


def save_results(output_path: str, results: list[dict], mode: str = "w"):
    """Save results to CSV file."""
    if not results:
        return

    fieldnames = ["qid", "predicted_answer", "answer"]
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(results)


async def run_simple_qa_benchmark(model_id: str):
    """Run Simple QA benchmark for a specific model."""
    # Setup paths
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"
    dataset_path = script_dir.parent / "user_data" / "simple_qa_test_subset_qid.csv"
    output_path = script_dir / f"{model_id}_simple_qa_prediction.csv"

    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))

    print(f"Getting model config for: {model_id}")
    model_config = get_model_config(config, model_id)

    print(f"Creating LLM client...")
    llm_client = create_llm_client(model_config["params"]["model"])

    if llm_client is None:
        raise ValueError("Failed to create LLM client")

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(str(dataset_path))
    print(f"Loaded {len(dataset)} questions")

    # Process dataset
    all_results = []
    batch_results = []

    for idx, item in enumerate(dataset, 1):
        qid = item["qid"]
        problem = item["problem"]
        answer = item["answer"]

        print(f"Processing question {idx}/{len(dataset)} (qid: {qid})")

        try:
            # Generate prediction
            predicted_answer = await llm_client.generate(
                prompt=problem,
                system_prompt=model_config["params"].get("system_prompt", ""),
            )

            result = {
                "qid": qid,
                "predicted_answer": predicted_answer.strip(),
                "answer": answer,
            }

            batch_results.append(result)
            all_results.append(result)

            print(f"  Predicted: {predicted_answer.strip()[:100]}...")

            # Save every 10 predictions
            if idx % 10 == 0:
                print(f"Saving results at {idx} predictions...")
                if idx == 10:
                    # First batch - write with header
                    save_results(str(output_path), batch_results, mode="w")
                else:
                    # Append to existing file
                    save_results(str(output_path), batch_results, mode="a")
                batch_results = []

        except Exception as e:
            print(f"Error processing qid {qid}: {str(e)}")
            # Log error and continue
            result = {
                "qid": qid,
                "predicted_answer": f"ERROR: {str(e)}",
                "answer": answer,
            }
            batch_results.append(result)
            all_results.append(result)

    # Save remaining results
    if batch_results:
        print(f"Saving final batch of {len(batch_results)} results...")
        if len(all_results) <= 10:
            # If total is 10 or less, write with header
            save_results(str(output_path), batch_results, mode="w")
        else:
            # Append to existing file
            save_results(str(output_path), batch_results, mode="a")

    print(f"\nCompleted! Results saved to: {output_path}")
    print(f"Total questions processed: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Simple QA benchmark evaluation for a specific model"
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID from config.yaml (e.g., generator_gemini_2_5_flash_lite)",
    )

    args = parser.parse_args()

    # Run the async function
    asyncio.run(run_simple_qa_benchmark(args.model_id))


if __name__ == "__main__":
    main()
