"""Built-in evaluators for LLM output assessment."""

import json
import re


def exact_match(actual: str, expected: str, metadata: dict) -> dict:
    """
    Evaluate if actual output exactly matches expected (case-insensitive).

    Args:
        actual: The actual output from the model
        expected: The expected output
        metadata: Optional metadata (unused)

    Returns:
        Dictionary with 'passed' bool and optional 'score'
    """
    passed = actual.strip().lower() == expected.strip().lower()
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": f"{'Match' if passed else 'Mismatch'}: '{actual.strip()}' vs '{expected.strip()}'",
    }


def contains(actual: str, expected: str, metadata: dict) -> dict:
    """
    Evaluate if actual output contains the expected substring.

    Args:
        actual: The actual output from the model
        expected: The substring to find
        metadata: Optional metadata (unused)

    Returns:
        Dictionary with 'passed' bool
    """
    passed = expected.lower() in actual.lower()
    return {
        "passed": passed,
        "details": f"Expected substring {'found' if passed else 'not found'} in output",
    }


def semantic_similarity(actual: str, expected: str, metadata: dict) -> dict:
    """
    Evaluate semantic similarity between outputs (simple token overlap).

    Note: This is a lightweight similarity check based on token overlap.
    For production use, consider using sentence-transformers or embedding APIs.

    Args:
        actual: The actual output from the model
        expected: The expected output
        metadata: Optional metadata (unused)

    Returns:
        Dictionary with 'passed' bool and 'score'
    """
    # Simple token-based similarity
    actual_tokens = set(actual.lower().split())
    expected_tokens = set(expected.lower().split())

    if not expected_tokens:
        return {"passed": True, "score": 1.0, "details": "Empty expected output"}

    intersection = actual_tokens & expected_tokens
    union = actual_tokens | expected_tokens

    similarity = len(intersection) / len(union) if union else 0.0

    # Pass if similarity > 0.5
    passed = similarity > 0.5

    return {
        "passed": passed,
        "score": similarity,
        "details": f"Token overlap similarity: {similarity:.2%}",
    }


def json_valid(actual: str, expected: str, metadata: dict) -> dict:
    """
    Evaluate if actual output is valid JSON.

    Args:
        actual: The actual output from the model
        expected: The expected JSON (optional, for validation)
        metadata: Optional metadata (unused)

    Returns:
        Dictionary with 'passed' bool
    """
    try:
        json.loads(actual)
        passed = True
        details = "Valid JSON"
    except (json.JSONDecodeError, ValueError) as e:
        passed = False
        details = f"Invalid JSON: {str(e)[:100]}"

    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": details,
    }


def response_length(actual: str, expected: str, metadata: dict) -> dict:
    """
    Evaluate if actual output length is within specified bounds.

    Args:
        actual: The actual output from the model
        expected: The expected output (unused)
        metadata: Dictionary with optional 'min_length' and 'max_length'

    Returns:
        Dictionary with 'passed' bool and 'score'
    """
    actual_length = len(actual)
    min_length = metadata.get("min_length", 0)
    max_length = metadata.get("max_length", float("inf"))

    passed = min_length <= actual_length <= max_length

    score = 1.0
    if actual_length < min_length:
        score = actual_length / min_length if min_length > 0 else 0.0
    elif actual_length > max_length:
        score = max_length / actual_length if actual_length > 0 else 0.0

    return {
        "passed": passed,
        "score": score,
        "details": f"Length: {actual_length} (expected {min_length}-{max_length})",
    }


def toxicity_check(actual: str, expected: str, metadata: dict) -> dict:
    """
    Simple heuristic check for toxic content.

    This is a lightweight filter using pattern matching.
    For production use, consider using a specialized toxicity detection API.

    Args:
        actual: The actual output from the model
        expected: The expected output (unused)
        metadata: Optional metadata (unused)

    Returns:
        Dictionary with 'passed' bool
    """
    # Simple pattern-based detection
    toxic_patterns = [
        r"\b(hate|kill|destroy|harm|violent)\b",
        r"\b(racist|sexist|bigot)\b",
        r"\b(curse|damn|hell)\b",
    ]

    text_lower = actual.lower()
    found_toxic = False

    for pattern in toxic_patterns:
        if re.search(pattern, text_lower):
            found_toxic = True
            break

    passed = not found_toxic

    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": f"{'No toxic patterns detected' if passed else 'Toxic content detected'}",
    }


def custom_regex(actual: str, expected: str, metadata: dict) -> dict:
    """
    Evaluate if actual output matches a regex pattern.

    Args:
        actual: The actual output from the model
        expected: The expected output (unused)
        metadata: Dictionary with required 'pattern' key (regex string)

    Returns:
        Dictionary with 'passed' bool

    Raises:
        ValueError: If 'pattern' not in metadata
    """
    pattern = metadata.get("pattern")
    if not pattern:
        raise ValueError("'pattern' required in metadata for custom_regex evaluator")

    try:
        passed = bool(re.search(pattern, actual))
    except re.error as e:
        return {
            "passed": False,
            "details": f"Invalid regex pattern: {str(e)}",
        }

    return {
        "passed": passed,
        "details": f"Pattern {'matched' if passed else 'not matched'}",
    }


def fuzzy_match(actual: str, expected: str, metadata: dict) -> dict:
    """
    Evaluate using simple fuzzy string matching.

    Args:
        actual: The actual output from the model
        expected: The expected output
        metadata: Dictionary with optional 'threshold' (default 0.8)

    Returns:
        Dictionary with 'passed' bool and 'score'
    """
    threshold = metadata.get("threshold", 0.8)

    # Simple character-based similarity
    actual_clean = actual.strip().lower()
    expected_clean = expected.strip().lower()

    # Longest common subsequence length divided by max length
    common = sum(1 for a, e in zip(actual_clean, expected_clean) if a == e)
    max_len = max(len(actual_clean), len(expected_clean))

    similarity = common / max_len if max_len > 0 else 0.0
    passed = similarity >= threshold

    return {
        "passed": passed,
        "score": similarity,
        "details": f"Fuzzy match similarity: {similarity:.2%}",
    }
