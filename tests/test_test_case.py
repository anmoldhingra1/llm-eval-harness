"""Tests for test_case module."""

import pytest
from llm_eval.test_case import TestCase, EvalResult, EvalResults
from llm_eval import evaluators


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_test_case_creation(self):
        """Test creating a test case."""
        tc = TestCase(
            input="hello",
            expected_output="world",
            evaluator=evaluators.exact_match
        )
        assert tc.input == "hello"
        assert tc.expected_output == "world"
        assert tc.evaluator == evaluators.exact_match

    def test_test_case_with_metadata(self):
        """Test test case with metadata."""
        metadata = {"category": "test", "priority": 1}
        tc = TestCase(
            input="hello",
            expected_output="world",
            evaluator=evaluators.exact_match,
            metadata=metadata
        )
        assert tc.metadata == metadata

    def test_test_case_empty_input_raises(self):
        """Test empty input raises ValueError."""
        with pytest.raises(ValueError, match="input cannot be empty"):
            TestCase(
                input="",
                expected_output="world",
                evaluator=evaluators.exact_match
            )

    def test_test_case_empty_expected_raises(self):
        """Test empty expected output raises ValueError."""
        with pytest.raises(ValueError, match="expected_output cannot be empty"):
            TestCase(
                input="hello",
                expected_output="",
                evaluator=evaluators.exact_match
            )

    def test_test_case_non_callable_evaluator_raises(self):
        """Test non-callable evaluator raises ValueError."""
        with pytest.raises(ValueError, match="evaluator must be callable"):
            TestCase(
                input="hello",
                expected_output="world",
                evaluator="not_callable"
            )

    def test_test_case_get_test_id(self):
        """Test get_test_id generates consistent ID."""
        tc = TestCase(
            input="test input",
            expected_output="output",
            evaluator=evaluators.exact_match
        )
        test_id = tc.get_test_id()
        assert test_id.startswith("test_")
        assert test_id == tc.get_test_id()  # Consistent

    def test_test_case_to_dict(self):
        """Test converting test case to dict."""
        tc = TestCase(
            input="hello",
            expected_output="world",
            evaluator=evaluators.exact_match,
            metadata={"key": "value"}
        )
        d = tc.to_dict()
        assert d["input"] == "hello"
        assert d["expected_output"] == "world"
        assert d["evaluator"] == "exact_match"
        assert d["metadata"] == {"key": "value"}


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_eval_result_creation(self):
        """Test creating an eval result."""
        result = EvalResult(
            test_id="test_1",
            passed=True,
            score=0.95,
            details="Test passed",
            actual_output="output",
            expected_output="expected",
            evaluator_name="exact_match"
        )
        assert result.test_id == "test_1"
        assert result.passed is True
        assert result.score == 0.95

    def test_eval_result_default_score(self):
        """Test eval result sets default score."""
        result = EvalResult(
            test_id="test_1",
            passed=True,
            evaluator_name="test"
        )
        assert result.score == 1.0

    def test_eval_result_failed_default_score(self):
        """Test eval result failed keeps score 0."""
        result = EvalResult(
            test_id="test_1",
            passed=False,
            evaluator_name="test"
        )
        assert result.score == 0.0

    def test_eval_result_to_dict(self):
        """Test converting result to dict."""
        result = EvalResult(
            test_id="test_1",
            passed=True,
            score=0.9,
            details="passed",
            evaluator_name="exact_match"
        )
        d = result.to_dict()
        assert d["test_id"] == "test_1"
        assert d["passed"] is True
        assert d["score"] == 0.9
        assert d["evaluator"] == "exact_match"


class TestEvalResults:
    """Tests for EvalResults dataclass."""

    def test_eval_results_creation(self):
        """Test creating eval results."""
        results = EvalResults(harness_name="test_harness")
        assert results.harness_name == "test_harness"
        assert len(results.results) == 0

    def test_eval_results_add_result(self):
        """Test adding results."""
        results = EvalResults(harness_name="test")
        result = EvalResult(
            test_id="test_1",
            passed=True,
            evaluator_name="test"
        )
        results.add_result(result)
        assert len(results.results) == 1
        assert results.results[0] == result

    def test_eval_results_add_multiple(self):
        """Test adding multiple results."""
        results = EvalResults(harness_name="test")
        for i in range(3):
            result = EvalResult(
                test_id=f"test_{i}",
                passed=i % 2 == 0,
                evaluator_name="test"
            )
            results.add_result(result)
        assert len(results.results) == 3

    def test_eval_results_get_summary_empty(self):
        """Test summary on empty results."""
        results = EvalResults(harness_name="test")
        summary = results.get_summary()
        assert summary["total"] == 0
        assert summary["passed"] == 0
        assert summary["failed"] == 0
        assert summary["pass_rate"] == 0.0
        assert summary["average_score"] == 0.0

    def test_eval_results_get_summary(self):
        """Test summary calculation."""
        results = EvalResults(harness_name="test")
        results.add_result(EvalResult(
            test_id="test_1",
            passed=True,
            score=1.0,
            evaluator_name="test"
        ))
        results.add_result(EvalResult(
            test_id="test_2",
            passed=False,
            score=0.0,
            evaluator_name="test"
        ))
        summary = results.get_summary()
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["average_score"] == 0.5

    def test_eval_results_get_summary_scores(self):
        """Test summary with varying scores."""
        results = EvalResults(harness_name="test")
        for score in [0.8, 0.9, 1.0]:
            results.add_result(EvalResult(
                test_id=f"test_{score}",
                passed=True,
                score=score,
                evaluator_name="test"
            ))
        summary = results.get_summary()
        assert summary["total"] == 3
        assert summary["passed"] == 3
        assert summary["average_score"] == pytest.approx(0.9)

    def test_eval_results_to_dict(self):
        """Test converting results to dict."""
        results = EvalResults(harness_name="test")
        results.add_result(EvalResult(
            test_id="test_1",
            passed=True,
            evaluator_name="test"
        ))
        d = results.to_dict()
        assert d["harness_name"] == "test"
        assert "summary" in d
        assert "results" in d
        assert len(d["results"]) == 1
