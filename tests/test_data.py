import pytest
from datasets import Dataset, DatasetDict

from text2cypher.data import REQUIRED_COLUMNS, validate_dataset


def make_valid_dataset() -> DatasetDict:
    example = {
        column: ["example"]
        for column in REQUIRED_COLUMNS
    }
    example["database_reference_alias"] = [None]

    return DatasetDict(
        {
            "train": Dataset.from_dict(example),
            "val": Dataset.from_dict(example),
            "test": Dataset.from_dict(example),
        }
    )


def test_validate_dataset_accepts_valid_dataset() -> None:
    dataset = make_valid_dataset()

    validate_dataset(dataset)


def test_validate_dataset_rejects_missing_split() -> None:
    dataset = make_valid_dataset()
    del dataset["test"]

    with pytest.raises(ValueError, match="missing required splits"):
        validate_dataset(dataset)


def test_validate_dataset_rejects_missing_column() -> None:
    dataset = make_valid_dataset()
    dataset["train"] = dataset["train"].remove_columns("cypher")

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(dataset)