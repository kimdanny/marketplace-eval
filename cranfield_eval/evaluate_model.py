import argparse
import asyncio
import csv
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.llm_client import create_llm_client
from prompts.simple_qa_judge_prompt import GRADER_TEMPLATE


def load_predictions(prediction_path: str) -> dict[str, dict]:
    """Load predictions from CSV file."""
    predictions = {}
    with open(prediction_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions[row["qid"]] = {
                "predicted_answer": row["predicted_answer"],
                "answer": row["answer"],
            }
    return predictions


def load_original_dataset(dataset_path: str) -> dict[str, dict]:
    """Load the original dataset."""
    dataset = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset[row["qid"]] = {"problem": row["problem"], "answer": row["answer"]}
    return dataset


def save_grades(output_path: str, grades: list[dict], mode: str = "w"):
    """Save grades to CSV file."""
    if not grades:
        return

    fieldnames = ["qid", "grade"]
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(grades)


def extract_grade(response: str) -> str:
    """Extract grade (A, B, or C) from LLM response."""
    # Clean up the response
    response = response.strip().upper()

    # Look for A, B, or C in the response
    if len(response) == 1 and response in ["A", "B", "C"]:
        return response
    else:
        # Default to D if unclear
        print(
            f"Warning: Could not extract grade from response: '{response}'. Defaulting to D."
        )
        return "D"


async def evaluate_model(model_id: str):
    """Evaluate model predictions using LLM judge."""
    # Setup paths
    script_dir = Path(__file__).parent
    prediction_path = script_dir / f"{model_id}_simple_qa_prediction.csv"
    dataset_path = script_dir.parent / "user_data" / "simple_qa_test_subset_qid.csv"
    output_path = script_dir / f"{model_id}_judged.csv"

    # Check if prediction file exists
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_path}\n"
            f"Please run: python run_simple_qa_benchmark.py {model_id}"
        )

    print(f"Loading predictions from: {prediction_path}")
    predictions = load_predictions(str(prediction_path))
    print(f"Loaded {len(predictions)} predictions")

    print(f"Loading original dataset from: {dataset_path}")
    dataset = load_original_dataset(str(dataset_path))
    print(f"Loaded {len(dataset)} questions")

    # Create LLM judge client
    print("Creating LLM judge client (openai/gpt-4.1)...")
    judge_config = {
        "provider": "openrouter",
        "model_id": "openai/gpt-4.1",
        "generation_parameters": {
            "temperature": 0.0,  # Use 0 for consistent grading
        },
    }
    judge_client = create_llm_client(judge_config)

    if judge_client is None:
        raise ValueError("Failed to create LLM judge client")

    # Grade predictions
    all_grades = []
    batch_grades = []

    # Sort by qid for consistent ordering
    sorted_qids = sorted(predictions.keys(), key=int)

    for idx, qid in enumerate(sorted_qids, 1):
        prediction = predictions[qid]
        question_data = dataset.get(qid)

        if not question_data:
            print(f"Warning: Question {qid} not found in original dataset. Skipping.")
            continue

        print(f"Grading {idx}/{len(sorted_qids)} (qid: {qid})")

        try:
            # Format the grader prompt
            grader_prompt = GRADER_TEMPLATE.format(
                question=question_data["problem"],
                target=question_data["answer"],
                predicted_answer=prediction["predicted_answer"],
            )

            # Get grade from judge
            response = await judge_client.generate(prompt=grader_prompt)
            grade = extract_grade(response)

            grade_result = {"qid": qid, "grade": grade}

            batch_grades.append(grade_result)
            all_grades.append(grade_result)

            print(f"  Grade: {grade}")

            # Save every 10 grades
            if idx % 10 == 0:
                print(f"Saving grades at {idx} evaluations...")
                if idx == 10:
                    # First batch - write with header
                    save_grades(str(output_path), batch_grades, mode="w")
                else:
                    # Append to existing file
                    save_grades(str(output_path), batch_grades, mode="a")
                batch_grades = []

        except Exception as e:
            print(f"Error grading qid {qid}: {str(e)}")
            # Log error and continue with a default grade
            grade_result = {
                "qid": qid,
                "grade": "C",  # Default to NOT_ATTEMPTED on error
            }
            batch_grades.append(grade_result)
            all_grades.append(grade_result)

    # Save remaining grades
    if batch_grades:
        print(f"Saving final batch of {len(batch_grades)} grades...")
        if len(all_grades) <= 10:
            # If total is 10 or less, write with header
            save_grades(str(output_path), batch_grades, mode="w")
        else:
            # Append to existing file
            save_grades(str(output_path), batch_grades, mode="a")

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Total questions evaluated: {len(all_grades)}")

    # Count grades
    grade_counts = {"A": 0, "B": 0, "C": 0}
    for grade_result in all_grades:
        grade_counts[grade_result["grade"]] += 1

    print(f"\nGrade Distribution:")
    print(
        f"  A (CORRECT):       {grade_counts['A']:4d} ({grade_counts['A']/len(all_grades)*100:.1f}%)"
    )
    print(
        f"  B (INCORRECT):     {grade_counts['B']:4d} ({grade_counts['B']/len(all_grades)*100:.1f}%)"
    )
    print(
        f"  C (NOT_ATTEMPTED): {grade_counts['C']:4d} ({grade_counts['C']/len(all_grades)*100:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions using LLM judge"
    )
    parser.add_argument(
        "model_id", type=str, help="Model ID (e.g., generator_gemini_2_5_flash_lite)"
    )

    args = parser.parse_args()

    # Run the async function
    asyncio.run(evaluate_model(args.model_id))


if __name__ == "__main__":
    main()
