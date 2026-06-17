import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from text2cypher.config import MODEL_NAME
from text2cypher.data import load_text2cypher_dataset
from text2cypher.evaluation import (
    evaluate_sample,
    generate_cypher,
    save_results,
    summarize_results,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("results/evaluation.json")
DEFAULT_MAX_NEW_TOKENS = 192


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a text-to-Cypher language model."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face model identifier or local checkpoint path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for summary and per-sample evaluation results.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of generated tokens per sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample limit for smoke tests.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of PyTorch CPU threads.",
    )

    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.num_threads < 1:
        raise ValueError("--num-threads must be at least 1")

    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")


def main() -> None:
    args = parse_args()
    configure_logging()
    validate_args(args)

    torch.set_num_threads(args.num_threads)

    logger.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()

    dataset = load_text2cypher_dataset()
    evaluation_dataset = dataset[args.split]

    if args.max_samples is not None:
        evaluation_dataset = evaluation_dataset.select(
            range(min(args.max_samples, len(evaluation_dataset)))
        )

    logger.info(
        "Evaluating %d samples from split '%s'",
        len(evaluation_dataset),
        args.split,
    )

    results: list[dict] = []

    for index, example in enumerate(evaluation_dataset, start=1):
        prediction = generate_cypher(
            model=model,
            tokenizer=tokenizer,
            schema=example["schema"],
            question=example["question"],
            max_new_tokens=args.max_new_tokens,
        )

        result = evaluate_sample(
            example=example,
            prediction=prediction,
        )
        results.append(result)

        logger.info(
            "Evaluated %d/%d: exact_match=%.0f, token_f1=%.3f",
            index,
            len(evaluation_dataset),
            result["normalized_exact_match"],
            result["token_f1"],
        )

    summary = summarize_results(results)

    save_results(
        output_path=args.output_path,
        model_path=args.model_path,
        split=args.split,
        summary=summary,
        samples=results,
    )

    logger.info("Evaluation summary: %s", summary)
    logger.info("Saved results to %s", args.output_path)


if __name__ == "__main__":
    main()