import pytest
import torch
from transformers import AutoTokenizer

from text2cypher.collator import CausalLMCollator
from text2cypher.config import IGNORE_INDEX, MODEL_NAME


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def test_collator_dynamically_pads_examples(tokenizer) -> None:
    collator = CausalLMCollator(tokenizer)

    examples = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [IGNORE_INDEX, 2, 3],
        },
        {
            "input_ids": [4, 5],
            "attention_mask": [1, 1],
            "labels": [IGNORE_INDEX, 5],
        },
    ]

    batch = collator(examples)

    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)


def test_collator_uses_ignore_index_for_label_padding(tokenizer) -> None:
    collator = CausalLMCollator(tokenizer)

    examples = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [IGNORE_INDEX, 2, 3],
        },
        {
            "input_ids": [4],
            "attention_mask": [1],
            "labels": [5],
        },
    ]

    batch = collator(examples)

    assert batch["labels"][1].tolist() == [
        5,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_collator_attention_mask_marks_padding(tokenizer) -> None:
    collator = CausalLMCollator(tokenizer)

    examples = [
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [1, 2],
        },
        {
            "input_ids": [3],
            "attention_mask": [1],
            "labels": [3],
        },
    ]

    batch = collator(examples)

    assert batch["attention_mask"][0].tolist() == [1, 1]
    assert batch["attention_mask"][1].tolist() == [1, 0]


def test_collator_returns_long_tensors(tokenizer) -> None:
    collator = CausalLMCollator(tokenizer)

    batch = collator(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "labels": [IGNORE_INDEX, 2],
            }
        ]
    )

    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_collator_rejects_empty_batch(tokenizer) -> None:
    collator = CausalLMCollator(tokenizer)

    with pytest.raises(ValueError, match="examples must not be empty"):
        collator([])