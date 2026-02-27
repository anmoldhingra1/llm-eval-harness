"""Tests for evaluators module."""

import pytest
from llm_eval import evaluators


class TestExactMatch:
    """Tests for exact_match evaluator."""

    def test_exact_match_pass(self):
        """Test exact match passes for identical strings."""
        result = evaluators.exact_match("hello", "hello", {})
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_exact_match_case_insensitive(self):
        """Test exact match is case-insensitive."""
        result = evaluators.exact_match("HELLO", "hello", {})
        assert result["passed"] is True

    def test_exact_match_with_whitespace(self):
        """Test exact match handles whitespace."""
        result = evaluators.exact_match("  hello  ", "hello", {})
        assert result["passed"] is True

    def test_exact_match_fail(self):
        """Test exact match fails for different strings."""
        result = evaluators.exact_match("hello", "world", {})
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_exact_match_has_details(self):
        """Test exact match includes details."""
        result = evaluators.exact_match("hello", "world", {})
        assert "details" in result
        assert "Mismatch" in result["details"]


class TestContains:
    """Tests for contains evaluator."""

    def test_contains_pass(self):
        """Test contains passes when substring found."""
        result = evaluators.contains("hello world", "hello", {})
        assert result["passed"] is True

    def test_contains_case_insensitive(self):
        """Test contains is case-insensitive."""
        result = evaluators.contains("HELLO WORLD", "hello", {})
        assert result["passed"] is True

    def test_contains_fail(self):
        """Test contains fails when substring not found."""
        result = evaluators.contains("hello", "world", {})
        assert result["passed"] is False

    def test_contains_partial_match(self):
        """Test contains with partial match."""
        result = evaluators.contains("the quick brown fox", "brown", {})
        assert result["passed"] is True


class TestSemanticSimilarity:
    """Tests for semantic_similarity evaluator."""

    def test_semantic_similarity_high_overlap(self):
        """Test semantic similarity with high token overlap."""
        result = evaluators.semantic_similarity(
            "the cat sat on the mat",
            "the cat sat on the mat",
            {}
        )
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_semantic_similarity_partial_overlap(self):
        """Test semantic similarity with partial overlap."""
        result = evaluators.semantic_similarity(
            "the cat sat on the mat",
            "the dog sat on the mat",
            {}
        )
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_semantic_similarity_no_overlap(self):
        """Test semantic similarity with no overlap."""
        result = evaluators.semantic_similarity(
            "apple banana orange",
            "zebra xylophone yak",
            {}
        )
        assert result["score"] == 0.0

    def test_semantic_similarity_empty_expected(self):
        """Test semantic similarity with empty expected output."""
        result = evaluators.semantic_similarity("hello", "", {})
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_semantic_similarity_threshold(self):
        """Test semantic similarity passes above threshold."""
        result = evaluators.semantic_similarity(
            "the cat sat on the mat",
            "the cat sat on the mat",
            {}
        )
        assert result["passed"] is True


class TestJsonValid:
    """Tests for json_valid evaluator."""

    def test_json_valid_pass(self):
        """Test json_valid passes for valid JSON."""
        result = evaluators.json_valid('{"key": "value"}', "", {})
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_json_valid_array(self):
        """Test json_valid with JSON array."""
        result = evaluators.json_valid('[1, 2, 3]', "", {})
        assert result["passed"] is True

    def test_json_valid_number(self):
        """Test json_valid with JSON number."""
        result = evaluators.json_valid('42', "", {})
        assert result["passed"] is True

    def test_json_valid_fail(self):
        """Test json_valid fails for invalid JSON."""
        result = evaluators.json_valid('{"key": invalid}', "", {})
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_json_valid_malformed(self):
        """Test json_valid with malformed JSON."""
        result = evaluators.json_valid("{key: value}", "", {})
        assert result["passed"] is False

    def test_json_valid_has_details(self):
        """Test json_valid includes error details."""
        result = evaluators.json_valid("not json", "", {})
        assert "details" in result


class TestResponseLength:
    """Tests for response_length evaluator."""

    def test_response_length_within_bounds(self):
        """Test response length within bounds."""
        result = evaluators.response_length(
            "hello world",
            "",
            {"min_length": 5, "max_length": 20}
        )
        assert result["passed"] is True

    def test_response_length_too_short(self):
        """Test response length too short."""
        result = evaluators.response_length(
            "hi",
            "",
            {"min_length": 5, "max_length": 20}
        )
        assert result["passed"] is False
        assert result["score"] < 1.0

    def test_response_length_too_long(self):
        """Test response length too long."""
        result = evaluators.response_length(
            "a" * 100,
            "",
            {"min_length": 5, "max_length": 20}
        )
        assert result["passed"] is False

    def test_response_length_defaults(self):
        """Test response length with defaults."""
        result = evaluators.response_length("hello", "", {})
        assert result["passed"] is True

    def test_response_length_exact(self):
        """Test response length at exact boundary."""
        result = evaluators.response_length(
            "hello",
            "",
            {"min_length": 5, "max_length": 5}
        )
        assert result["passed"] is True


class TestToxicityCheck:
    """Tests for toxicity_check evaluator."""

    def test_toxicity_check_clean(self):
        """Test toxicity check on clean content."""
        result = evaluators.toxicity_check("This is a nice message", "", {})
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_toxicity_check_hate_word(self):
        """Test toxicity check detects hate."""
        result = evaluators.toxicity_check("I hate this", "", {})
        assert result["passed"] is False

    def test_toxicity_check_violent_word(self):
        """Test toxicity check detects violent content."""
        result = evaluators.toxicity_check("This will destroy everything", "", {})
        assert result["passed"] is False

    def test_toxicity_check_curse_word(self):
        """Test toxicity check detects curse words."""
        result = evaluators.toxicity_check("What the hell is this", "", {})
        assert result["passed"] is False

    def test_toxicity_check_case_insensitive(self):
        """Test toxicity check is case-insensitive."""
        result = evaluators.toxicity_check("This is HATE", "", {})
        assert result["passed"] is False

    def test_toxicity_check_has_details(self):
        """Test toxicity check includes details."""
        result = evaluators.toxicity_check("I love this", "", {})
        assert "details" in result


class TestCustomRegex:
    """Tests for custom_regex evaluator."""

    def test_custom_regex_match(self):
        """Test custom regex matches pattern."""
        result = evaluators.custom_regex(
            "my email is test@example.com",
            "",
            {"pattern": r".*@.*\.com"}
        )
        assert result["passed"] is True

    def test_custom_regex_no_match(self):
        """Test custom regex no match."""
        result = evaluators.custom_regex(
            "my name is john",
            "",
            {"pattern": r".*@.*\.com"}
        )
        assert result["passed"] is False

    def test_custom_regex_phone(self):
        """Test custom regex for phone numbers."""
        result = evaluators.custom_regex(
            "Call me at 555-1234",
            "",
            {"pattern": r"\d{3}-\d{4}"}
        )
        assert result["passed"] is True

    def test_custom_regex_missing_pattern(self):
        """Test custom regex raises error without pattern."""
        with pytest.raises(ValueError, match="'pattern' required"):
            evaluators.custom_regex("text", "", {})

    def test_custom_regex_invalid_pattern(self):
        """Test custom regex with invalid regex pattern."""
        result = evaluators.custom_regex(
            "text",
            "",
            {"pattern": "[invalid("}
        )
        assert result["passed"] is False
        assert "Invalid regex pattern" in result["details"]


class TestFuzzyMatch:
    """Tests for fuzzy_match evaluator."""

    def test_fuzzy_match_exact(self):
        """Test fuzzy match with exact match."""
        result = evaluators.fuzzy_match("hello", "hello", {})
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_fuzzy_match_similar(self):
        """Test fuzzy match with similar strings."""
        result = evaluators.fuzzy_match("hello", "hallo", {})
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_fuzzy_match_threshold(self):
        """Test fuzzy match with custom threshold."""
        result = evaluators.fuzzy_match(
            "hello",
            "hello",
            {"threshold": 0.9}
        )
        assert result["passed"] is True

    def test_fuzzy_match_low_threshold(self):
        """Test fuzzy match with low threshold."""
        result = evaluators.fuzzy_match(
            "abcdef",
            "abcxyz",
            {"threshold": 0.5}
        )
        assert result["passed"] is True

    def test_fuzzy_match_case_insensitive(self):
        """Test fuzzy match is case-insensitive."""
        result = evaluators.fuzzy_match("HELLO", "hello", {})
        assert result["score"] == 1.0

    def test_fuzzy_match_with_whitespace(self):
        """Test fuzzy match strips whitespace."""
        result = evaluators.fuzzy_match("  hello  ", "hello", {})
        assert result["score"] == 1.0
