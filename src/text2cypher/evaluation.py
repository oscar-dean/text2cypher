import json
from pathlib import Path
from statistics import mean

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from text2cypher.config import MAX_LENGTH
from text2cypher.metrics import (
    canonicalized_exact_match,
    component_agreement,
    has_basic_query_structure,
    normalize_cypher,
    normalized_exact_match,
    token_multiset_prf,
)
from text2cypher.prompts import build_generation_messages


def clean_generated_query(text: str) -> str:
    """Remove surrounding whitespace and accidental Markdown formatting."""
    return normalize_cypher(text)


def generate_cypher(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    schema: str,
    question: str,
    max_new_tokens: int,
) -> str:
    """Generate one Cypher query using deterministic decoding."""
    messages = build_generation_messages(
        {
            "schema": schema,
            "question": question,
        }
    )

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    encoded = {
        name: tensor.to(model.device)
        for name, tensor in encoded.items()
    }

    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = generated[0, prompt_length:]
    prediction = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return clean_generated_query(prediction)


def evaluate_sample(example: dict, prediction: str) -> dict:
    """Compute all metrics for one generated query."""
    reference = example["cypher"]

    return {
        "instance_id": example["instance_id"],
        "data_source": example["data_source"],
        "database_reference_alias": example["database_reference_alias"],
        "schema": example["schema"],
        "question": example["question"],
        "ground_truth": reference,
        "prediction": prediction,
        "normalized_ground_truth": normalize_cypher(reference),
        "normalized_prediction": normalize_cypher(prediction),
        "normalized_exact_match": normalized_exact_match(
            prediction,
            reference,
        ),
        "canonicalized_exact_match": canonicalized_exact_match(
            prediction,
            reference,
        ),
        "has_basic_query_structure": float(
            has_basic_query_structure(prediction)
        ),
        **token_multiset_prf(prediction, reference),
        **component_agreement(prediction, reference),
    }


def summarize_results(results: list[dict]) -> dict[str, float | int]:
    """Aggregate per-sample metrics."""
    if not results:
        raise ValueError("results must not be empty")

    metric_names = [
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

    summary: dict[str, float | int] = {
        "num_samples": len(results),
    }

    for metric_name in metric_names:
        summary[metric_name] = mean(
            result[metric_name] for result in results
        )

    return summary


def save_results(
    output_path: Path,
    model_path: str,
    split: str,
    summary: dict[str, float | int],
    samples: list[dict],
) -> None:
    """Save summary and per-sample results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model_path,
        "split": split,
        "summary": summary,
        "samples": samples,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )