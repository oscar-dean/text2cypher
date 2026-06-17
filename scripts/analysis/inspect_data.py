from statistics import mean

from transformers import AutoTokenizer

from text2cypher.data import load_text2cypher_dataset
from text2cypher.prompts import SYSTEM_PROMPT, build_user_prompt

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


def build_training_messages(example: dict) -> list[dict[str, str]]:
    """Build a complete training conversation."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(
                schema=example["schema"],
                question=example["question"],
            ),
        },
        {
            "role": "assistant",
            "content": example["cypher"],
        },
    ]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_text2cypher_dataset()

    token_lengths: list[int] = []

    for example in dataset["train"]:
        messages = build_training_messages(example)

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )

        token_lengths.append(len(encoded["input_ids"]))
        
        # print(type(encoded))
        # print(encoded.keys())
        # print(type(encoded["input_ids"]))

    sorted_lengths = sorted(token_lengths)

    def percentile(value: float) -> int:
        index = min(
            int(value * len(sorted_lengths)),
            len(sorted_lengths) - 1,
        )
        return sorted_lengths[index]

    print(f"Examples: {len(token_lengths)}")
    print(f"Minimum: {min(token_lengths)}")
    print(f"Mean: {mean(token_lengths):.1f}")
    print(f"Median: {percentile(0.50)}")
    print(f"90th percentile: {percentile(0.90)}")
    print(f"95th percentile: {percentile(0.95)}")
    print(f"99th percentile: {percentile(0.99)}")
    print(f"Maximum: {max(token_lengths)}")

    first_example = dataset["train"][0]
    formatted_text = tokenizer.apply_chat_template(
        build_training_messages(first_example),
        tokenize=False,
        add_generation_prompt=False,
    )

    print("\nFirst formatted training example:\n")
    print(formatted_text)


if __name__ == "__main__":
    main()