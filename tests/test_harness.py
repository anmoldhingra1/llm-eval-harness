"""Tests for harness module."""

import pytest
from llm_eval import EvalHarness, TestCase
from llm_eval.test_case import EvalResults


class TestEvalHarness:
    """Tests for EvalHarness class."""

    def test_harness_creation(self):
        """Test creating a harness."""
        harness = EvalHarness(name="test_harness")
        assert harness.name == "test_harness"
        assert harness.description == ""
        assert len(harness.get_test_cases()) == 0

    def test_harness_with_description(self):
        """Test harness with description."""
        harness = EvalHarness(
            name="test",
            description="Test harness"
        )
        assert harness.description == "Test harness"

    def test_harness_add_test_case(self, sample_test_case):
        """Test adding a test case."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        assert len(harness.get_test_cases()) == 1

    def test_harness_add_invalid_test_case(self):
        """Test adding invalid test case raises error."""
        harness = EvalHarness(name="test")
        with pytest.raises(ValueError, match="TestCase instance"):
            harness.add_test_case("not a test case")

    def test_harness_add_multiple_test_cases(self, sample_test_cases):
        """Test adding multiple test cases."""
        harness = EvalHarness(name="test")
        harness.add_test_cases(sample_test_cases)
        assert len(harness.get_test_cases()) == len(sample_test_cases)

    def test_harness_get_test_cases(self, sample_test_case):
        """Test getting test cases returns copy."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        cases = harness.get_test_cases()
        assert len(cases) == 1
        # Modifying returned list shouldn't affect harness
        cases.clear()
        assert len(harness.get_test_cases()) == 1

    def test_harness_clear_test_cases(self, sample_test_case):
        """Test clearing test cases."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        assert len(harness.get_test_cases()) == 1
        harness.clear_test_cases()
        assert len(harness.get_test_cases()) == 0

    def test_harness_run_no_test_cases(self):
        """Test run raises error without test cases."""
        harness = EvalHarness(name="test")
        with pytest.raises(ValueError, match="No test cases registered"):
            harness.run({})

    def test_harness_run_basic(self, sample_test_case):
        """Test basic harness run."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        results = harness.run({"What is 2+2?": "4"})
        assert isinstance(results, EvalResults)
        assert results.harness_name == "test"
        assert len(results.results) == 1

    def test_harness_run_matching(self, sample_test_case):
        """Test run with matching output."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        results = harness.run({"What is 2+2?": "4"})
        assert results.results[0].passed is True

    def test_harness_run_non_matching(self, sample_test_case):
        """Test run with non-matching output."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        results = harness.run({"What is 2+2?": "5"})
        assert results.results[0].passed is False

    def test_harness_run_no_matching_input(self, sample_test_case):
        """Test run with no matching input."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        results = harness.run({"Different input": "output"})
        assert results.results[0].passed is False
        assert "No matching output" in results.results[0].details

    def test_harness_run_partial_match(self, sample_test_case):
        """Test run with partial substring match."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        # Contains the input as substring
        results = harness.run({
            "What is 2+2? This is a longer input": "4"
        })
        # Should still match due to substring fallback
        assert results.results[0].passed is True

    def test_harness_run_stores_results(self, sample_test_case):
        """Test run stores last results."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        results1 = harness.run({"What is 2+2?": "4"})
        assert harness.get_results() == results1

    def test_harness_run_overwrites_results(self, sample_test_case):
        """Test run overwrites previous results."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        harness.run({"What is 2+2?": "4"})
        results2 = harness.run({"What is 2+2?": "5"})
        assert harness.get_results() == results2
        assert harness.get_results().results[0].passed is False

    def test_harness_report_no_results(self):
        """Test report raises error without results."""
        harness = EvalHarness(name="test")
        with pytest.raises(ValueError, match="No evaluation results"):
            harness.report()

    def test_harness_report_terminal(self, sample_test_case):
        """Test terminal report generation."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        harness.run({"What is 2+2?": "4"})
        report = harness.report(format="terminal")
        assert isinstance(report, str)
        assert "test" in report
        assert "PASS" in report or "FAIL" in report

    def test_harness_report_json(self, sample_test_case):
        """Test JSON report generation."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        harness.run({"What is 2+2?": "4"})
        report = harness.report(format="json")
        assert isinstance(report, str)
        # Should be valid JSON
        import json
        data = json.loads(report)
        assert "harness_name" in data

    def test_harness_report_csv(self, sample_test_case):
        """Test CSV report generation."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        harness.run({"What is 2+2?": "4"})
        report = harness.report(format="csv")
        assert isinstance(report, str)
        assert "test_id" in report

    def test_harness_report_markdown(self, sample_test_case):
        """Test Markdown report generation."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        harness.run({"What is 2+2?": "4"})
        report = harness.report(format="markdown")
        assert isinstance(report, str)
        assert "#" in report  # Markdown headers

    def test_harness_report_invalid_format(self, sample_test_case):
        """Test report raises error for invalid format."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        harness.run({"What is 2+2?": "4"})
        with pytest.raises(ValueError, match="Unknown report format"):
            harness.report(format="invalid")

    def test_harness_get_results_none(self):
        """Test get_results returns None before run."""
        harness = EvalHarness(name="test")
        assert harness.get_results() is None

    def test_harness_summary_no_results(self):
        """Test summary raises error without results."""
        harness = EvalHarness(name="test")
        with pytest.raises(ValueError, match="No evaluation results"):
            harness.summary()

    def test_harness_summary(self, sample_test_case):
        """Test summary generation."""
        harness = EvalHarness(name="test")
        harness.add_test_case(sample_test_case)
        harness.run({"What is 2+2?": "4"})
        summary = harness.summary()
        assert isinstance(summary, dict)
        assert "total" in summary
        assert "passed" in summary
        assert "pass_rate" in summary

    def test_harness_multiple_test_cases(self, sample_test_cases, sample_outputs):
        """Test harness with multiple test cases."""
        harness = EvalHarness(name="multi_test")
        harness.add_test_cases(sample_test_cases)
        results = harness.run(sample_outputs)
        assert len(results.results) == len(sample_test_cases)

    def test_harness_evaluator_error_handling(self):
        """Test harness handles evaluator errors gracefully."""
        def bad_evaluator(actual, expected, metadata):
            raise RuntimeError("Evaluator error")

        harness = EvalHarness(name="test")
        tc = TestCase(
            input="test",
            expected_output="expected",
            evaluator=bad_evaluator
        )
        harness.add_test_case(tc)
        results = harness.run({"test": "actual"})
        assert results.results[0].passed is False
        assert "Evaluator error" in results.results[0].details

    def test_harness_evaluator_invalid_return(self):
        """Test harness handles invalid evaluator return."""
        def bad_evaluator(actual, expected, metadata):
            return "not a dict"

        harness = EvalHarness(name="test")
        tc = TestCase(
            input="test",
            expected_output="expected",
            evaluator=bad_evaluator
        )
        harness.add_test_case(tc)
        results = harness.run({"test": "actual"})
        assert results.results[0].passed is False

    def test_harness_evaluator_missing_passed_key(self):
        """Test harness handles evaluator missing 'passed' key."""
        def bad_evaluator(actual, expected, metadata):
            return {"score": 0.5}  # Missing 'passed'

        harness = EvalHarness(name="test")
        tc = TestCase(
            input="test",
            expected_output="expected",
            evaluator=bad_evaluator
        )
        harness.add_test_case(tc)
        results = harness.run({"test": "actual"})
        assert results.results[0].passed is False
