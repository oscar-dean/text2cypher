import json

import pytest

from text2cypher.evaluation import (
    clean_generated_query,
    evaluate_sample,
    save_results,
    summarize_results,
)


def sample_example() -> dict:
    return {
        "instance_id": "example-1",
        "data_source": "unit-test",
        "database_reference_alias": None,
        "schema": "Graph schema: Person {name: STRING}",
        "question": "Who is named Alice?",
        "cypher": "MATCH (p:Person {name: 'Alice'}) RETURN p",
    }


def test_clean_generated_query_removes_markdown() -> None:
    prediction = "```cypher\nMATCH (n) RETURN n;\n```"

    assert clean_generated_query(prediction) == "MATCH (n) RETURN n"


def test_evaluate_sample_contains_required_fields() -> None:
    example = sample_example()
    prediction = example["cypher"]

    result = evaluate_sample(example, prediction)

    assert result["instance_id"] == "example-1"
    assert result["question"] == example["question"]
    assert result["ground_truth"] == example["cypher"]
    assert result["prediction"] == prediction
    assert result["normalized_exact_match"] == 1.0
    assert result["canonicalized_exact_match"] == 1.0
    assert result["token_f1"] == pytest.approx(1.0)


def test_summarize_results_averages_metrics() -> None:
    exact_result = evaluate_sample(
        sample_example(),
        sample_example()["cypher"],
    )
    incorrect_result = evaluate_sample(
        sample_example(),
        "",
    )

    summary = summarize_results([exact_result, incorrect_result])

    assert summary["num_samples"] == 2
    assert summary["normalized_exact_match"] == pytest.approx(0.5)


def test_summarize_results_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="results must not be empty"):
        summarize_results([])


def test_save_results_writes_json(tmp_path) -> None:
    output_path = tmp_path / "evaluation.json"

    samples = [
        evaluate_sample(
            sample_example(),
            sample_example()["cypher"],
        )
    ]
    summary = summarize_results(samples)

    save_results(
        output_path=output_path,
        model_path="test-model",
        split="test",
        summary=summary,
        samples=samples,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["model"] == "test-model"
    assert payload["split"] == "test"
    assert payload["summary"]["num_samples"] == 1
    assert len(payload["samples"]) == 1