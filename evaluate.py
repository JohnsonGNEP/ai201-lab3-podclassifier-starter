import json
import os

from config import VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import classify_episode, load_labeled_examples


def run_evaluation() -> dict:
    """
    Run the classifier against the held-out test set and return full results.

    This function:
      1. Loads the labeled training examples from my_labels.json
      2. Loads the test episodes with ground-truth labels
      3. Runs classify_episode() on each test description
      4. Returns predictions, ground truth, and per-episode detail
    """
    labeled_examples = load_labeled_examples()

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)

    results = []
    for episode in test_episodes:
        print(f"  Classifying: {episode['title'][:60]}...")
        prediction = classify_episode(episode["description"], labeled_examples)
        results.append({
            "id": episode["id"],
            "title": episode["title"],
            "description": episode["description"],
            "ground_truth": episode["label"],
            "predicted": prediction["label"],
            "reasoning": prediction["reasoning"],
            "correct": prediction["label"] == episode["label"],
        })

    predictions = [r["predicted"] for r in results]
    ground_truth = [r["ground_truth"] for r in results]

    return {
        "results": results,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "total": len(results),
    }


def compute_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Compute overall classification accuracy.

    Accuracy is the number of exact prediction/ground-truth matches divided by
    the number of paired labels. Empty inputs return 0.0.
    """
    if not predictions or not ground_truth:
        return 0.0

    total = min(len(predictions), len(ground_truth))
    if total == 0:
        return 0.0

    correct = sum(
        1
        for predicted, actual in zip(predictions, ground_truth)
        if predicted == actual
    )
    return correct / total


def compute_per_class_accuracy(
    predictions: list[str], ground_truth: list[str]
) -> dict[str, dict]:
    """
    Compute accuracy broken down by each label class.

    For each label in VALID_LABELS, returns correct count, total ground-truth
    count, and accuracy. Classes with no ground-truth examples get accuracy 0.0.
    """
    stats = {
        label: {"correct": 0, "total": 0, "accuracy": 0.0}
        for label in VALID_LABELS
    }

    for predicted, actual in zip(predictions, ground_truth):
        if actual not in stats:
            continue

        stats[actual]["total"] += 1
        if predicted == actual:
            stats[actual]["correct"] += 1

    for label_stats in stats.values():
        total = label_stats["total"]
        if total:
            label_stats["accuracy"] = label_stats["correct"] / total

    return stats


def format_evaluation_report(eval_results: dict) -> str:
    """
    Format evaluation results into a readable report string.
    """
    predictions = eval_results["predictions"]
    ground_truth = eval_results["ground_truth"]
    results = eval_results["results"]

    accuracy = compute_accuracy(predictions, ground_truth)
    per_class = compute_per_class_accuracy(predictions, ground_truth)

    lines = [
        f"## Evaluation Results\n",
        f"**Overall accuracy:** {accuracy:.1%} ({sum(r['correct'] for r in results)}/{eval_results['total']})\n",
        "\n**Per-class accuracy:**",
    ]
    for label, stats in per_class.items():
        bar = "#" * int(stats["accuracy"] * 10) + "-" * (10 - int(stats["accuracy"] * 10))
        lines.append(f"  {label:<12} {bar}  {stats['accuracy']:.0%}  ({stats['correct']}/{stats['total']})")

    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        lines.append(f"\n**Misclassified ({len(misclassified)}):**")
        for r in misclassified:
            lines.append(f"  [{r['ground_truth']} -> {r['predicted']}] {r['title']}")
    else:
        lines.append("\n**No misclassifications - perfect score!**")

    return "\n".join(lines)
