SYSTEM_PROMPT = (
    "You translate natural-language questions into Cypher queries. "
    "Use only node labels, relationship types, and properties provided in the graph schema. "
    "Return only the Cypher query without explanation or Markdown formatting."
)


def build_user_prompt(schema: str, question: str) -> str:
    """
    Build the user message for text-to-Cypher generation.

    Args:
        schema: Graph schema describing available nodes, relationships, and properties.
        question: Natural-language question to translate.

    Returns:
        Formatted user message containing the schema and question.
    """
    schema = schema.strip()
    question = question.strip()

    if not schema:
        raise ValueError("schema must not be empty")

    if not question:
        raise ValueError("question must not be empty")

    return f"{schema}\n\nQuestion:\n{question}"


def build_training_messages(example: dict) -> list[dict[str, str]]:
    """Build the complete system-user-assistant conversation for SFT."""
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


def build_generation_messages(example: dict) -> list[dict[str, str]]:
    """Build the system-user conversation used during generation."""
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
    ]