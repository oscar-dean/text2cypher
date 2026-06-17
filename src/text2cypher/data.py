from datasets import DatasetDict, load_dataset, Dataset
from transformers import PreTrainedTokenizerBase
from text2cypher.config import IGNORE_INDEX, MAX_LENGTH
from text2cypher.prompts import (
    build_generation_messages,
    build_training_messages,
)

DATASET_NAME = "RomanTeucher/text2cypher-curated"
REQUIRED_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {
    "question",
    "schema",
    "cypher",
    "data_source",
    "instance_id",
    "database_reference_alias",
}


def validate_dataset(dataset: DatasetDict) -> None:
    """
    Validate that the dataset contains all required splits and columns.

    Args:
        dataset: Dataset to validate.

    Raises:
        ValueError: If required splits or columns are missing.
    """
    missing_splits = REQUIRED_SPLITS - set(dataset.keys())
    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {sorted(missing_splits)}")

    for split_name in REQUIRED_SPLITS:
        split_columns = set(dataset[split_name].column_names)
        missing_columns = REQUIRED_COLUMNS - split_columns

        if missing_columns:
            raise ValueError(
                f"Split '{split_name}' is missing required columns: "
                f"{sorted(missing_columns)}"
            )


def load_text2cypher_dataset(
    dataset_name: str = DATASET_NAME,
) -> DatasetDict:
    """
    Load and validate the text-to-Cypher dataset.

    Args:
        dataset_name: Hugging Face dataset identifier.

    Returns:
        Validated dataset with train, validation, and test splits.
    """
    dataset = load_dataset(dataset_name)
    validate_dataset(dataset)
    return dataset


def tokenize_training_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_LENGTH,
) -> dict[str, list[int]]:
    """
    Tokenize one SFT example and mask prompt tokens from the loss.

    Only assistant-response tokens contribute to the training loss.
    """
    prompt_messages = build_generation_messages(example)
    full_messages = build_training_messages(example)

    prompt_encoding = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    full_encoding = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    input_ids = full_encoding["input_ids"]
    attention_mask = full_encoding["attention_mask"]

    prompt_length = len(prompt_encoding["input_ids"])

    labels = input_ids.copy()

    # Ignore the system prompt, schema, question, and assistant prefix.
    labels[:prompt_length] = [IGNORE_INDEX] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_LENGTH,
) -> DatasetDict:
    """Tokenize all dataset splits for supervised fine-tuning."""

    def tokenize_example(example: dict) -> dict[str, list[int]]:
        return tokenize_training_example(
            example=example,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_example,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )


def count_fully_masked_examples(dataset: Dataset) -> int:
    """Count examples whose labels contain no trainable target tokens."""
    return sum(
        all(label == IGNORE_INDEX for label in example["labels"])
        for example in dataset
    )