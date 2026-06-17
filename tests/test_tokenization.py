from transformers import AutoTokenizer

from text2cypher.config import IGNORE_INDEX, MODEL_NAME
from text2cypher.data import tokenize_training_example


def sample_example() -> dict:
    return {
        "schema": "Graph schema: Person {name: STRING}",
        "question": "Who is named Alice?",
        "cypher": "MATCH (p:Person {name: 'Alice'}) RETURN p",
    }


def test_tokenize_training_example_masks_prompt_tokens() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encoded = tokenize_training_example(sample_example(), tokenizer)

    labels = encoded["labels"]

    assert len(encoded["input_ids"]) == len(labels)
    assert len(encoded["attention_mask"]) == len(labels)
    assert IGNORE_INDEX in labels
    assert any(label != IGNORE_INDEX for label in labels)


def test_tokenize_training_example_respects_max_length() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encoded = tokenize_training_example(
        sample_example(),
        tokenizer,
        max_length=32,
    )

    assert len(encoded["input_ids"]) <= 32