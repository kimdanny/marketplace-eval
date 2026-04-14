import argparse
import csv
import json
from pathlib import Path


def load_grades(judged_path: str) -> list[str]:
    """Load grades from judged CSV file."""
    grades = []
    with open(judged_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grades.append(row["grade"])
    return grades


def calculate_metrics(grades: list[str]) -> dict:
    """Calculate metrics from grades."""
    total = len(grades)

    if total == 0:
        raise ValueError("No grades found in file")

    # Count each grade
    count_a = sum(1 for g in grades if g == "A")
    count_b = sum(1 for g in grades if g == "B")
    count_c = sum(1 for g in grades if g == "C")

    # Calculate proportions
    is_correct = count_a / total
    is_incorrect = count_b / total
    is_not_attempted = count_c / total

    # Given attempted = correct + incorrect (i.e., A + B)
    is_given_attempted = is_correct + is_incorrect

    # Accuracy given attempted (correct among those that attempted)
    accuracy_given_attempted = (
        is_correct / is_given_attempted if is_given_attempted > 0 else 0
    )

    # F1 Score (harmonic mean of accuracy_given_attempted and is_correct)
    f1_score = (
        2
        * accuracy_given_attempted
        * is_correct
        / (accuracy_given_attempted + is_correct)
        if (accuracy_given_attempted + is_correct) > 0
        else 0
    )

    return {
        "total_questions": total,
        "count_correct": count_a,
        "count_incorrect": count_b,
        "count_not_attempted": count_c,
        "is_correct": is_correct,
        "is_incorrect": is_incorrect,
        "is_not_attempted": is_not_attempted,
        "is_given_attempted": is_given_attempted,
        "accuracy_given_attempted": accuracy_given_attempted,
        "f1_score": f1_score,
    }


def print_metrics(model_id: str, metrics: dict):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Metrics for: {model_id}")
    print(f"{'='*60}")
    print(f"\nTotal Questions: {metrics['total_questions']}")
    print(f"\nGrade Counts:")
    print(
        f"  A (CORRECT):       {metrics['count_correct']:4d} ({metrics['is_correct']*100:5.1f}%)"
    )
    print(
        f"  B (INCORRECT):     {metrics['count_incorrect']:4d} ({metrics['is_incorrect']*100:5.1f}%)"
    )
    print(
        f"  C (NOT_ATTEMPTED): {metrics['count_not_attempted']:4d} ({metrics['is_not_attempted']*100:5.1f}%)"
    )
    print(f"\n{'='*60}")
    print(f"AGGREGATE METRICS")
    print(f"{'='*60}")
    print(
        f"  Correct Rate:              {metrics['is_correct']:.4f} ({metrics['is_correct']*100:.2f}%)"
    )
    print(
        f"  Incorrect Rate:            {metrics['is_incorrect']:.4f} ({metrics['is_incorrect']*100:.2f}%)"
    )
    print(
        f"  Not Attempted Rate:        {metrics['is_not_attempted']:.4f} ({metrics['is_not_attempted']*100:.2f}%)"
    )
    print(
        f"  Given Attempted Rate:      {metrics['is_given_attempted']:.4f} ({metrics['is_given_attempted']*100:.2f}%)"
    )
    print(f"\n{'='*60}")
    print(f"KEY METRICS")
    print(f"{'='*60}")
    print(
        f"  Accuracy Given Attempted:  {metrics['accuracy_given_attempted']:.4f} ({metrics['accuracy_given_attempted']*100:.2f}%)"
    )
    print(
        f"  F1 Score:                  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)"
    )
    print(f"{'='*60}\n")


def save_metrics(output_path: str, model_id: str, metrics: dict):
    """Save metrics to JSON file."""
    output_data = {"model_id": model_id, "metrics": metrics}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Metrics saved to: {output_path}")


def get_metrics(model_id: str, save: bool = False):
    """Calculate and display metrics for a model."""
    script_dir = Path(__file__).parent
    judged_path = script_dir / f"{model_id}_judged.csv"

    if not judged_path.exists():
        raise FileNotFoundError(
            f"Judged file not found: {judged_path}\n"
            f"Please run: python evaluate_model.py {model_id}"
        )

    # Load grades
    grades = load_grades(str(judged_path))

    # Calculate metrics
    metrics = calculate_metrics(grades)

    # Print metrics
    print_metrics(model_id, metrics)

    # Optionally save to JSON
    if save:
        output_path = script_dir / f"{model_id}_metrics.json"
        save_metrics(str(output_path), model_id, metrics)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics from judged results"
    )
    parser.add_argument(
        "model_id",
        type=str,
        nargs="?",
        help="Model ID (e.g., generator_gemini_2_5_flash_lite). If not provided, calculates for all models.",
    )
    parser.add_argument("--save", action="store_true", help="Save metrics to JSON file")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Calculate metrics for all models with judged files",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # If --all or no model_id provided, find all judged files
    if args.all or args.model_id is None:
        judged_files = list(script_dir.glob("*_judged.csv"))

        if not judged_files:
            print("No judged files found!")
            print("Please run evaluate_model.py first.")
            return

        # Extract model IDs
        model_ids = []
        for judged_file in judged_files:
            model_id = judged_file.stem.replace("_judged", "")
            model_ids.append(model_id)

        # Calculate metrics for all models
        all_metrics = {}
        for model_id in sorted(model_ids):
            try:
                metrics = get_metrics(model_id, save=args.save)
                all_metrics[model_id] = metrics
            except Exception as e:
                print(f"Error processing {model_id}: {str(e)}")

        # Print summary comparison
        if len(all_metrics) > 1:
            print(f"\n{'='*80}")
            print(f"SUMMARY COMPARISON")
            print(f"{'='*80}")
            print(f"{'Model':<45} {'Correct %':>10} {'F1 Score':>10} {'Acc@Att':>10}")
            print(f"{'-'*80}")
            for model_id, metrics in sorted(all_metrics.items()):
                print(
                    f"{model_id:<45} {metrics['is_correct']*100:>9.2f}% {metrics['f1_score']*100:>9.2f}% {metrics['accuracy_given_attempted']*100:>9.2f}%"
                )
            print(f"{'='*80}\n")
    else:
        # Calculate for single model
        get_metrics(args.model_id, save=args.save)


if __name__ == "__main__":
    main()
