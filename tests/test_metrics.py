import pytest

from text2cypher.metrics import (
    canonicalize_variable_names,
    canonicalized_exact_match,
    component_agreement,
    extract_cypher_components,
    has_basic_query_structure,
    normalize_cypher,
    normalized_exact_match,
    token_multiset_prf,
    tokenize_cypher,
)


def test_normalize_cypher_removes_markdown_and_semicolon() -> None:
    query = "```cypher\nMATCH (n) RETURN n;\n```"

    assert normalize_cypher(query) == "MATCH (n) RETURN n"


def test_normalized_exact_match_ignores_whitespace() -> None:
    prediction = "MATCH   (n)\nRETURN n;"
    reference = "MATCH (n) RETURN n"

    assert normalized_exact_match(prediction, reference) == 1.0


def test_normalized_exact_match_rejects_different_variable_names() -> None:
    prediction = "MATCH (person:Person) RETURN person.name"
    reference = "MATCH (p:Person) RETURN p.name"

    assert normalized_exact_match(prediction, reference) == 0.0


def test_tokenize_cypher_preserves_string_literal() -> None:
    tokens = tokenize_cypher(
        "MATCH (p:Person {name: 'Christopher Nolan'}) RETURN p"
    )

    assert "'Christopher Nolan'" in tokens
    assert "MATCH" in tokens
    assert "Person" in tokens


def test_token_multiset_prf_identical_queries() -> None:
    metrics = token_multiset_prf(
        "MATCH (n) RETURN n",
        "MATCH (n) RETURN n",
    )

    assert metrics["token_precision"] == pytest.approx(1.0)
    assert metrics["token_recall"] == pytest.approx(1.0)
    assert metrics["token_f1"] == pytest.approx(1.0)


def test_token_multiset_prf_empty_prediction() -> None:
    metrics = token_multiset_prf(
        "",
        "MATCH (n) RETURN n",
    )

    assert metrics == {
        "token_precision": 0.0,
        "token_recall": 0.0,
        "token_f1": 0.0,
    }


def test_token_multiset_f1_can_be_high_for_different_queries() -> None:
    prediction = "MATCH (a)-[:DIRECTED]->(b) RETURN b"
    reference = "MATCH (b)<-[:DIRECTED]-(a) RETURN b"

    metrics = token_multiset_prf(prediction, reference)

    assert metrics["token_f1"] > 0.8
    assert normalized_exact_match(prediction, reference) == 0.0


def test_basic_query_structure_requires_match_and_return() -> None:
    assert has_basic_query_structure("MATCH (n) RETURN n") is True
    assert has_basic_query_structure("MATCH (n)") is False
    assert has_basic_query_structure("") is False


def test_canonicalize_variable_names_uses_declaration_order() -> None:
    query = (
        "MATCH (person:Person)-[relation:DIRECTED]->(movie:Movie) "
        "WHERE person.name = 'Christopher Nolan' "
        "RETURN movie.title"
    )

    canonicalized = canonicalize_variable_names(query)

    assert canonicalized == (
        "MATCH (v0:Person)-[v1:DIRECTED]->(v2:Movie) "
        "WHERE v0.name = 'Christopher Nolan' "
        "RETURN v2.title"
    )


def test_canonicalized_exact_match_ignores_variable_names() -> None:
    prediction = (
        "MATCH (person:Person)-[:DIRECTED]->(movie:Movie) "
        "RETURN movie.title"
    )
    reference = (
        "MATCH (p:Person)-[:DIRECTED]->(m:Movie) "
        "RETURN m.title"
    )

    assert canonicalized_exact_match(prediction, reference) == 1.0


def test_canonicalized_exact_match_keeps_semantic_differences() -> None:
    prediction = (
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) "
        "RETURN m.title"
    )
    reference = (
        "MATCH (p:Person)-[:DIRECTED]->(m:Movie) "
        "RETURN m.title"
    )

    assert canonicalized_exact_match(prediction, reference) == 0.0


def test_extract_cypher_components() -> None:
    query = (
        "MATCH (p:Person {name: 'Christopher Nolan'})"
        "-[:DIRECTED]->(m:Movie) "
        "WHERE m.year < 2010 "
        "RETURN m.title"
    )

    components = extract_cypher_components(query)

    assert components.node_labels == frozenset({"Person", "Movie"})
    assert components.relationship_types == frozenset({"DIRECTED"})
    assert components.properties == frozenset({"name", "year", "title"})
    assert components.directions == ("->",)
    assert components.comparison_operators == ("<",)


def test_component_agreement_identical_components() -> None:
    prediction = (
        "MATCH (person:Person)-[:DIRECTED]->(movie:Movie) "
        "WHERE movie.year < 2010 "
        "RETURN movie.title"
    )
    reference = (
        "MATCH (p:Person)-[:DIRECTED]->(m:Movie) "
        "WHERE m.year < 2010 "
        "RETURN m.title"
    )

    metrics = component_agreement(prediction, reference)

    assert metrics["node_labels_match"] == 1.0
    assert metrics["relationship_types_match"] == 1.0
    assert metrics["properties_match"] == 1.0
    assert metrics["directions_match"] == 1.0
    assert metrics["comparison_operators_match"] == 1.0
    assert metrics["component_match_rate"] == pytest.approx(1.0)


def test_component_agreement_detects_wrong_relationship() -> None:
    prediction = (
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) "
        "RETURN m.title"
    )
    reference = (
        "MATCH (p:Person)-[:DIRECTED]->(m:Movie) "
        "RETURN m.title"
    )

    metrics = component_agreement(prediction, reference)

    assert metrics["node_labels_match"] == 1.0
    assert metrics["relationship_types_match"] == 0.0
    assert metrics["properties_match"] == 1.0
    assert metrics["directions_match"] == 1.0
    assert metrics["component_match_rate"] == pytest.approx(0.8)


def test_component_agreement_detects_wrong_direction() -> None:
    prediction = (
        "MATCH (p:Person)<-[:DIRECTED]-(m:Movie) "
        "RETURN m.title"
    )
    reference = (
        "MATCH (p:Person)-[:DIRECTED]->(m:Movie) "
        "RETURN m.title"
    )

    metrics = component_agreement(prediction, reference)

    assert metrics["directions_match"] == 0.0


def test_tokenize_incoming_relationship_direction() -> None:
    tokens = tokenize_cypher(
        "MATCH (a)<-[:DIRECTED]-(b) RETURN a"
    )

    assert "<-" in tokens
    assert "<" not in tokens


def test_extract_incoming_direction_is_not_comparison() -> None:
    components = extract_cypher_components(
        "MATCH (a)<-[:DIRECTED]-(b) RETURN a"
    )

    assert components.directions == ("<-",)
    assert components.comparison_operators == ()