"""Tests for report module."""

import json
import pytest
from llm_eval.report import ReportGenerator
from llm_eval.test_case import EvalResults, EvalResult


@pytest.fixture
def sample_results():
    """Create sample eval results for testing."""
    results = EvalResults(harness_name="test_harness")
    results.add_result(EvalResult(
        test_id="test_1",
        passed=True,
        score=1.0,
        details="Test passed",
        actual_output="output1",
        expected_output="expected1",
        evaluator_name="exact_match"
    ))
    results.add_result(EvalResult(
        test_id="test_2",
        passed=False,
        score=0.5,
        details="Test failed",
        actual_output="output2",
        expected_output="expected2",
        evaluator_name="semantic_similarity"
    ))
    return results


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_generate_terminal_report(self, sample_results):
        """Test terminal report generation."""
        report = ReportGenerator.generate_terminal_report(sample_results)
        assert isinstance(report, str)
        assert "test_harness" in report
        assert "PASS" in report or "FAIL" in report
        assert "test_1" in report
        assert "test_2" in report

    def test_terminal_report_has_summary(self, sample_results):
        """Test terminal report includes summary."""
        report = ReportGenerator.generate_terminal_report(sample_results)
        assert "Total Tests" in report
        assert "Passed" in report
        assert "Failed" in report
        assert "Pass Rate" in report

    def test_terminal_report_metrics(self, sample_results):
        """Test terminal report includes metrics."""
        report = ReportGenerator.generate_terminal_report(sample_results)
        # Should contain numbers from summary
        assert "2" in report  # total
        assert "1" in report  # passed

    def test_generate_json_report(self, sample_results):
        """Test JSON report generation."""
        report = ReportGenerator.generate_json_report(sample_results)
        assert isinstance(report, str)
        data = json.loads(report)
        assert data["harness_name"] == "test_harness"
        assert "summary" in data
        assert "results" in data

    def test_json_report_structure(self, sample_results):
        """Test JSON report has correct structure."""
        report = ReportGenerator.generate_json_report(sample_results)
        data = json.loads(report)
        summary = data["summary"]
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1

    def test_json_report_results(self, sample_results):
        """Test JSON report includes results."""
        report = ReportGenerator.generate_json_report(sample_results)
        data = json.loads(report)
        assert len(data["results"]) == 2
        assert data["results"][0]["test_id"] == "test_1"
        assert data["results"][0]["passed"] is True

    def test_generate_csv_report(self, sample_results):
        """Test CSV report generation."""
        report = ReportGenerator.generate_csv_report(sample_results)
        assert isinstance(report, str)
        lines = report.split("\n")
        assert len(lines) >= 2  # Header + data
        assert "test_id" in lines[0]

    def test_csv_report_header(self, sample_results):
        """Test CSV report has correct header."""
        report = ReportGenerator.generate_csv_report(sample_results)
        lines = report.split("\n")
        header = lines[0]
        assert "test_id" in header
        assert "passed" in header
        assert "score" in header
        assert "evaluator" in header

    def test_csv_report_data_rows(self, sample_results):
        """Test CSV report has data rows."""
        report = ReportGenerator.generate_csv_report(sample_results)
        lines = report.split("\n")
        assert len(lines) >= 3  # Header + 2 results
        assert "test_1" in lines[1]
        assert "test_2" in lines[2]

    def test_csv_report_escaping(self):
        """Test CSV report properly escapes quotes."""
        results = EvalResults(harness_name="test")
        results.add_result(EvalResult(
            test_id="test_1",
            passed=True,
            details='This has "quotes"',
            evaluator_name="test"
        ))
        report = ReportGenerator.generate_csv_report(results)
        assert '""' in report  # Escaped quotes

    def test_generate_markdown_report(self, sample_results):
        """Test Markdown report generation."""
        report = ReportGenerator.generate_markdown_report(sample_results)
        assert isinstance(report, str)
        assert "#" in report  # Markdown headers
        assert "test_harness" in report

    def test_markdown_report_structure(self, sample_results):
        """Test Markdown report has correct structure."""
        report = ReportGenerator.generate_markdown_report(sample_results)
        assert "# Evaluation Report" in report
        assert "## Summary" in report
        assert "## Results" in report

    def test_markdown_report_table(self, sample_results):
        """Test Markdown report has summary table."""
        report = ReportGenerator.generate_markdown_report(sample_results)
        assert "|" in report  # Table format
        assert "Total Tests" in report or "Metric" in report

    def test_markdown_report_pass_fail_status(self, sample_results):
        """Test Markdown report includes pass/fail status."""
        report = ReportGenerator.generate_markdown_report(sample_results)
        assert "PASS" in report or "✓" in report
        assert "FAIL" in report or "✗" in report

    def test_markdown_report_results(self, sample_results):
        """Test Markdown report includes all results."""
        report = ReportGenerator.generate_markdown_report(sample_results)
        assert "test_1" in report
        assert "test_2" in report
        assert "exact_match" in report
        assert "semantic_similarity" in report

    def test_report_empty_results(self):
        """Test reports with empty results."""
        results = EvalResults(harness_name="empty")

        terminal = ReportGenerator.generate_terminal_report(results)
        assert "empty" in terminal
        assert "0" in terminal

        json_report = ReportGenerator.generate_json_report(results)
        data = json.loads(json_report)
        assert data["summary"]["total"] == 0

        csv = ReportGenerator.generate_csv_report(results)
        lines = csv.split("\n")
        assert len(lines) == 1  # Just header

    def test_report_single_result(self):
        """Test reports with single result."""
        results = EvalResults(harness_name="single")
        results.add_result(EvalResult(
            test_id="test_1",
            passed=True,
            score=0.95,
            evaluator_name="test"
        ))

        json_report = ReportGenerator.generate_json_report(results)
        data = json.loads(json_report)
        assert data["summary"]["total"] == 1
        assert data["summary"]["pass_rate"] == 1.0

    def test_report_all_passed(self):
        """Test reports when all tests pass."""
        results = EvalResults(harness_name="all_pass")
        for i in range(3):
            results.add_result(EvalResult(
                test_id=f"test_{i}",
                passed=True,
                score=1.0,
                evaluator_name="test"
            ))

        json_report = ReportGenerator.generate_json_report(results)
        data = json.loads(json_report)
        assert data["summary"]["pass_rate"] == 1.0
        assert data["summary"]["passed"] == 3
        assert data["summary"]["failed"] == 0

    def test_report_all_failed(self):
        """Test reports when all tests fail."""
        results = EvalResults(harness_name="all_fail")
        for i in range(3):
            results.add_result(EvalResult(
                test_id=f"test_{i}",
                passed=False,
                score=0.0,
                evaluator_name="test"
            ))

        json_report = ReportGenerator.generate_json_report(results)
        data = json.loads(json_report)
        assert data["summary"]["pass_rate"] == 0.0
        assert data["summary"]["passed"] == 0
        assert data["summary"]["failed"] == 3

    def test_report_with_no_details(self):
        """Test reports with results lacking details."""
        results = EvalResults(harness_name="no_details")
        results.add_result(EvalResult(
            test_id="test_1",
            passed=True,
            evaluator_name="test"
            # No details
        ))

        terminal = ReportGenerator.generate_terminal_report(results)
        assert "test_1" in terminal

        json_report = ReportGenerator.generate_json_report(results)
        data = json.loads(json_report)
        assert data["results"][0]["details"] == ""
