import pytest

from text2cypher.prompts import build_user_prompt


def test_build_user_prompt_contains_schema_and_question() -> None:
    prompt = build_user_prompt(
        schema="Person {name: STRING}",
        question="Who is named Alice?",
    )

    assert "Person {name: STRING}" in prompt
    assert "Who is named Alice?" in prompt


def test_build_user_prompt_rejects_empty_schema() -> None:
    with pytest.raises(ValueError, match="schema must not be empty"):
        build_user_prompt("", "Who is named Alice?")


def test_build_user_prompt_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        build_user_prompt("Person {name: STRING}", "")

def test_build_user_prompt_does_not_duplicate_schema_heading() -> None:
    prompt = build_user_prompt(
        schema="Graph schema: Person {name: STRING}",
        question="Who is named Alice?",
    )

    assert prompt.count("Graph schema:") == 1