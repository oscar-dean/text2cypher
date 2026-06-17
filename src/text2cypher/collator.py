from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from text2cypher.config import IGNORE_INDEX


@dataclass
class CausalLMCollator:
    """
    Dynamically pad tokenized causal-language-model examples.

    Input IDs are padded with the tokenizer's padding token.
    Attention masks are padded with zero.
    Labels are padded with IGNORE_INDEX so padding does not affect the loss.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer must define either a pad token or an EOS token."
                )

            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(
        self,
        examples: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("examples must not be empty")

        input_features = [
            {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            for example in examples
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch_length = batch["input_ids"].shape[1]

        padded_labels = []

        for example in examples:
            labels = example["labels"]
            padding_length = batch_length - len(labels)

            if padding_length < 0:
                raise ValueError(
                    "Label sequence is longer than the padded input sequence."
                )

            padded_labels.append(
                labels + [IGNORE_INDEX] * padding_length
            )

        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long,
        )

        return batch