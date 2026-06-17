import argparse
import json
from pathlib import Path


METRICS = [
    "normalized_exact_match",
    "canonicalized_exact_match",
    "has_basic_query_structure",
    "token_precision",
    "token_recall",
    "token_f1",
    "node_labels_match",
    "relationship_types_match",
    "properties_match",
    "directions_match",
    "comparison_operators_match",
    "component_match_rate",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a Markdown comparison of two evaluation files."
    )
    parser.add_argument(
        "--base-results",
        type=Path,
        default=Path("results/base_model_test.json"),
        help="Path to the base-model evaluation JSON.",
    )
    parser.add_argument(
        "--fine-tuned-results",
        type=Path,
        default=Path("results/fine_tuned_model_test.json"),
        help="Path to the fine-tuned-model evaluation JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/model_comparison.md"),
        help="Path for the generated Markdown report.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of side-by-side examples to include.",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict:
    """Load an evaluation result file."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def format_percentage(value: float, show_sign: bool = False) -> str:
    """Format a fractional metric value as a percentage."""
    if show_sign:
        return f"{value * 100:+.2f}%"

    return f"{value * 100:.2f}%"


def index_samples(payload: dict) -> dict[str, dict]:
    """Index evaluation samples by instance ID."""
    return {
        sample["instance_id"]: sample
        for sample in payload["samples"]
    }


def build_report(
    base: dict,
    fine_tuned: dict,
    num_examples: int,
) -> str:
    """Build the complete Markdown comparison report."""
    if base["split"] != fine_tuned["split"]:
        raise ValueError("Evaluation files use different dataset splits.")

    lines = [
        "# Base vs Fine-Tuned Model",
        "",
        f"- **Base model:** `{base['model']}`",
        f"- **Fine-tuned model:** `{fine_tuned['model']}`",
        f"- **Evaluation split:** `{base['split']}`",
        "",
        "## Metric comparison",
        "",
        "| Metric | Base | Fine-tuned | Delta |",
        "|---|---:|---:|---:|",
    ]

    for metric in METRICS:
        base_value = float(base["summary"][metric])
        fine_value = float(fine_tuned["summary"][metric])
        delta = fine_value - base_value

        lines.append(
            f"| `{metric}` "
            f"| {format_percentage(base_value)} "
            f"| {format_percentage(fine_value)} "
            f"| {format_percentage(delta, show_sign=True)} |"
        )

    base_samples = index_samples(base)
    fine_samples = index_samples(fine_tuned)

    shared_ids = [
        instance_id
        for instance_id in base_samples
        if instance_id in fine_samples
    ]

    lines.extend(
        [
            "",
            "## Example comparisons",
            "",
        ]
    )

    for index, instance_id in enumerate(
        shared_ids[:num_examples],
        start=1,
    ):
        base_sample = base_samples[instance_id]
        fine_sample = fine_samples[instance_id]

        schema = base_sample.get("schema", "").strip()

        if not schema:
            schema = "Schema not available in evaluation result."

        lines.extend(
            [
                f"### Example {index}",
                "",
                f"**Instance:** `{instance_id}`",
                "",
                "**Graph schema**",
                "",
                "```text",
                schema,
                "```",
                "",
                "**Natural-language question**",
                "",
                base_sample["question"],
                "",
                "**Ground-truth Cypher**",
                "",
                "```cypher",
                base_sample["ground_truth"],
                "```",
                "",
                "**Base-model prediction**",
                "",
                "```text",
                base_sample["prediction"],
                "```",
                "",
                "**Fine-tuned-model prediction**",
                "",
                "```cypher",
                fine_sample["prediction"],
                "```",
                "",
                "| Model | Exact match | Token F1 | Component match |",
                "|---|---:|---:|---:|",
                (
                    "| Base "
                    f"| {format_percentage(base_sample['normalized_exact_match'])} "
                    f"| {format_percentage(base_sample['token_f1'])} "
                    f"| {format_percentage(base_sample['component_match_rate'])} |"
                ),
                (
                    "| Fine-tuned "
                    f"| {format_percentage(fine_sample['normalized_exact_match'])} "
                    f"| {format_percentage(fine_sample['token_f1'])} "
                    f"| {format_percentage(fine_sample['component_match_rate'])} |"
                ),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Generate the Markdown comparison report."""
    args = parse_args()

    if args.num_examples < 1:
        raise ValueError("--num-examples must be at least 1")

    base = load_results(args.base_results)
    fine_tuned = load_results(args.fine_tuned_results)

    report = build_report(
        base=base,
        fine_tuned=fine_tuned,
        num_examples=args.num_examples,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(f"Comparison written to {args.output}")


if __name__ == "__main__":
    main()