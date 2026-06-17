from transformers import AutoTokenizer

from text2cypher.config import MAX_LENGTH, MODEL_NAME
from text2cypher.data import (
    count_fully_masked_examples,
    load_text2cypher_dataset,
    tokenize_dataset,
)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_text2cypher_dataset()
    tokenized = tokenize_dataset(dataset, tokenizer)

    for split_name, split in tokenized.items():
        lengths = [len(example["input_ids"]) for example in split]

        print(
            f"{split_name}: "
            f"examples={len(split)}, "
            f"max_length={max(lengths)}, "
            f"fully_masked={count_fully_masked_examples(split)}"
        )

        assert max(lengths) <= MAX_LENGTH


if __name__ == "__main__":
    main()