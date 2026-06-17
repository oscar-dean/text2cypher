import re
from collections import Counter
from dataclasses import dataclass


CYPHER_TOKEN_PATTERN = re.compile(
    r"""
    '(?:\\.|[^'])*'          |
    "(?:\\.|[^"])*"          |
    ->|<-                    |
    <=|>=|<>|!=|=|<|>        |
    [A-Za-z_][A-Za-z0-9_]*   |
    \d+(?:\.\d+)?            |
    [()[\]{},.:;+\-*/]
    """,
    re.VERBOSE,
)

NODE_PATTERN = re.compile(
    r"""
    \(
    \s*
    (?P<variable>[A-Za-z_][A-Za-z0-9_]*)?
    \s*
    (?:
        :
        \s*
        (?P<label>[A-Za-z_][A-Za-z0-9_]*)
    )?
    """,
    re.VERBOSE,
)

RELATIONSHIP_PATTERN = re.compile(
    r"""
    \[
    \s*
    (?P<variable>[A-Za-z_][A-Za-z0-9_]*)?
    \s*
    (?:
        :
        \s*
        (?P<type>[A-Za-z_][A-Za-z0-9_]*)
    )?
    """,
    re.VERBOSE,
)

PROPERTY_ACCESS_PATTERN = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b"
)

PROPERTY_MAP_PATTERN = re.compile(r"\{(?P<content>[^{}]*)\}")

PROPERTY_MAP_KEY_PATTERN = re.compile(
    r"(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:"
)

KEYWORDS = {
    "AND",
    "AS",
    "ASC",
    "BY",
    "CALL",
    "CASE",
    "CREATE",
    "DELETE",
    "DESC",
    "DETACH",
    "DISTINCT",
    "ELSE",
    "END",
    "EXISTS",
    "IN",
    "IS",
    "LIMIT",
    "MATCH",
    "MERGE",
    "NOT",
    "NULL",
    "OPTIONAL",
    "OR",
    "ORDER",
    "REMOVE",
    "RETURN",
    "SET",
    "SKIP",
    "THEN",
    "UNION",
    "UNWIND",
    "WHEN",
    "WHERE",
    "WITH",
    "XOR",
}


@dataclass(frozen=True)
class CypherComponents:
    """Components extracted from a Cypher query for diagnostic comparison."""

    node_labels: frozenset[str]
    relationship_types: frozenset[str]
    properties: frozenset[str]
    directions: tuple[str, ...]
    comparison_operators: tuple[str, ...]


def normalize_cypher(query: str) -> str:
    """
    Normalize superficial formatting differences in a Cypher query.

    Removes Markdown fences, leading/trailing whitespace, a trailing
    semicolon, and repeated whitespace. Keyword case and query structure
    are preserved.
    """
    normalized = query.strip()

    if normalized.startswith("```"):
        normalized = re.sub(
            r"^```(?:cypher)?\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"\s*```$", "", normalized)

    normalized = normalized.strip().removesuffix(";").strip()
    return " ".join(normalized.split())


def normalized_exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when normalized queries are identical, otherwise 0.0."""
    return float(normalize_cypher(prediction) == normalize_cypher(reference))


def tokenize_cypher(query: str) -> list[str]:
    """Split a Cypher query into lexical tokens."""
    return CYPHER_TOKEN_PATTERN.findall(normalize_cypher(query))


def token_multiset_prf(
    prediction: str,
    reference: str,
) -> dict[str, float]:
    """
    Compute token-multiset precision, recall, and F1.

    Token order is ignored, while repeated token counts are preserved.
    This measures lexical overlap, not semantic correctness.
    """
    prediction_tokens = Counter(tokenize_cypher(prediction))
    reference_tokens = Counter(tokenize_cypher(reference))

    if not prediction_tokens and not reference_tokens:
        return {
            "token_precision": 1.0,
            "token_recall": 1.0,
            "token_f1": 1.0,
        }

    if not prediction_tokens:
        return {
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
        }

    overlap = sum((prediction_tokens & reference_tokens).values())
    predicted_count = sum(prediction_tokens.values())
    reference_count = sum(reference_tokens.values())

    precision = overlap / predicted_count
    recall = overlap / reference_count if reference_count else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def _collect_declared_variables(query: str) -> list[str]:
    """
    Collect node and relationship variables in order of first declaration.
    """
    matches: list[tuple[int, str]] = []

    for match in NODE_PATTERN.finditer(query):
        variable = match.group("variable")
        if variable:
            matches.append((match.start(), variable))

    for match in RELATIONSHIP_PATTERN.finditer(query):
        variable = match.group("variable")
        if variable:
            matches.append((match.start(), variable))

    ordered_variables: list[str] = []
    seen: set[str] = set()

    for _, variable in sorted(matches, key=lambda item: item[0]):
        if variable not in seen:
            seen.add(variable)
            ordered_variables.append(variable)

    return ordered_variables


def canonicalize_variable_names(query: str) -> str:
    """
    Replace declared Cypher variables with deterministic names.

    Variables are renamed by first declaration order:
    v0, v1, v2, ...

    The function does not reorder clauses or predicates and therefore
    only removes differences caused by arbitrary variable naming.
    """
    normalized = normalize_cypher(query)
    variables = _collect_declared_variables(normalized)

    variable_map = {
        variable: f"v{index}"
        for index, variable in enumerate(variables)
    }

    canonicalized = normalized

    # Replace longer names first to avoid accidental partial replacements.
    for variable in sorted(variable_map, key=len, reverse=True):
        canonical_name = variable_map[variable]
        canonicalized = re.sub(
            rf"\b{re.escape(variable)}\b",
            canonical_name,
            canonicalized,
        )

    return canonicalized


def canonicalized_exact_match(prediction: str, reference: str) -> float:
    """
    Compare queries after normalization and variable-name canonicalization.
    """
    return float(
        canonicalize_variable_names(prediction)
        == canonicalize_variable_names(reference)
    )


def has_basic_query_structure(query: str) -> bool:
    """
    Check for minimal Cypher-like structure.

    This is a diagnostic heuristic and does not prove syntactic or
    semantic correctness.
    """
    normalized = normalize_cypher(query).upper()

    if not normalized:
        return False

    return "MATCH" in normalized and "RETURN" in normalized


def extract_cypher_components(query: str) -> CypherComponents:
    """
    Extract selected Cypher components for diagnostic comparison.

    This is intentionally heuristic. It does not implement a full Cypher
    parser and may not cover every valid Cypher construct.
    """
    normalized = normalize_cypher(query)

    node_labels = {
        match.group("label")
        for match in NODE_PATTERN.finditer(normalized)
        if match.group("label")
    }

    relationship_types = {
        match.group("type")
        for match in RELATIONSHIP_PATTERN.finditer(normalized)
        if match.group("type")
    }

    accessed_properties = {
        property_name
        for _, property_name in PROPERTY_ACCESS_PATTERN.findall(normalized)
    }

    mapped_properties: set[str] = set()

    for property_map_match in PROPERTY_MAP_PATTERN.finditer(normalized):
        property_map_content = property_map_match.group("content")

        mapped_properties.update(
            key
            for key in PROPERTY_MAP_KEY_PATTERN.findall(property_map_content)
            if key.upper() not in KEYWORDS
        )

    directions = tuple(
        token
        for token in tokenize_cypher(normalized)
        if token in {"->", "<-"}
    )

    comparison_operators = tuple(
        token
        for token in tokenize_cypher(normalized)
        if token in {"<=", ">=", "<>", "!=", "=", "<", ">"}
    )

    return CypherComponents(
        node_labels=frozenset(node_labels),
        relationship_types=frozenset(relationship_types),
        properties=frozenset(accessed_properties | mapped_properties),
        directions=directions,
        comparison_operators=comparison_operators,
    )


def component_agreement(
    prediction: str,
    reference: str,
) -> dict[str, float]:
    """
    Compare selected Cypher components between two queries.

    Each returned value is 1.0 for an exact component match and 0.0
    otherwise. These values are diagnostic rather than semantic proof.
    """
    predicted = extract_cypher_components(prediction)
    expected = extract_cypher_components(reference)

    agreement = {
        "node_labels_match": float(
            predicted.node_labels == expected.node_labels
        ),
        "relationship_types_match": float(
            predicted.relationship_types == expected.relationship_types
        ),
        "properties_match": float(
            predicted.properties == expected.properties
        ),
        "directions_match": float(
            predicted.directions == expected.directions
        ),
        "comparison_operators_match": float(
            predicted.comparison_operators
            == expected.comparison_operators
        ),
    }

    agreement["component_match_rate"] = (
        sum(agreement.values()) / len(agreement)
    )

    return agreement