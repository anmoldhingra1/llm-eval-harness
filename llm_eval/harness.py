from __future__ import annotations

"""Main evaluation harness for LLM testing."""

from typing import Literal

from llm_eval.test_case import TestCase, EvalResult, EvalResults
from llm_eval.report import ReportGenerator


class EvalHarness:
    """

    Evaluation harness for testing LLM outputs.

    The harness manages test cases, runs evaluations against model outputs,
    and generates structured reports.

    Example:
        harness = EvalHarness(name="chatbot_eval")
        harness.add_test_case(TestCase(...))
        results = harness.run({"input": "output"})
        harness.report()
    """

    def __init__(self, name: str, description: str = "") -> None:
        """
        Initialize the evaluation harness.

        Args:
            name: Name of the evaluation harness
            description: Optional description of what this harness evaluates
        """
        self.name = name
        self.description = description
        self._test_cases: list[TestCase] = []
        self._last_results: EvalResults | None = None

    def add_test_case(self, test_case: TestCase) -> None:
        """
        Add a test case to the harness.

        Args:
            test_case: TestCase to add

        Raises:
            ValueError: If test case is invalid
        """
        if not isinstance(test_case, TestCase):
            raise ValueError("test_case must be a TestCase instance")
        self._test_cases.append(test_case)

    def add_test_cases(self, test_cases: list[TestCase]) -> None:
        """
        Add multiple test cases to the harness.

        Args:
            test_cases: List of TestCase objects to add
        """
        for test_case in test_cases:
            self.add_test_case(test_case)

    def get_test_cases(self) -> list[TestCase]:
        """
        Get all registered test cases.

        Returns:
            List of TestCase objects
        """
        return self._test_cases.copy()

    def clear_test_cases(self) -> None:
        """Clear all registered test cases."""
        self._test_cases = []

    def run(self, outputs: dict[str, str]) -> EvalResults:
        """
        Run evaluation against model outputs.

        Args:
            outputs: Dictionary mapping inputs to model outputs

        Returns:
            EvalResults object with detailed results

        Raises:
            ValueError: If test cases not found for outputs or vice versa
        """
        if not self._test_cases:
            raise ValueError("No test cases registered. Add test cases with add_test_case()")

        results = EvalResults(harness_name=self.name)

        for test_case in self._test_cases:
            # Find matching output for this test case
            actual_output = None
            for input_text, output_text in outputs.items():
                if input_text.strip() == test_case.input.strip():
                    actual_output = output_text
                    break

            if actual_output is None:
                # Fallback: try to match by substring
                for input_text, output_text in outputs.items():
                    if test_case.input in input_text or input_text in test_case.input:
                        actual_output = output_text
                        break

            if actual_output is None:
                # No matching output found, mark as failed
                result = EvalResult(
                    test_id=test_case.get_test_id(),
                    passed=False,
                    score=0.0,
                    details="No matching output found",
                    actual_output="",
                    expected_output=test_case.expected_output,
                    evaluator_name=test_case.evaluator.__name__,
                )
            else:
                # Run the evaluator
                try:
                    eval_output = test_case.evaluator(
                        actual_output,
                        test_case.expected_output,
                        test_case.metadata,
                    )

                    # Validate evaluator output
                    if not isinstance(eval_output, dict):
                        raise ValueError(
                            f"Evaluator {test_case.evaluator.__name__} must return dict"
                        )
                    if "passed" not in eval_output:
                        raise ValueError(
                            f"Evaluator {test_case.evaluator.__name__} must return 'passed' key"
                        )

                    passed = eval_output["passed"]
                    score = eval_output.get("score", 1.0 if passed else 0.0)
                    details = eval_output.get("details", "")

                    result = EvalResult(
                        test_id=test_case.get_test_id(),
                        passed=passed,
                        score=score,
                        details=details,
                        actual_output=actual_output,
                        expected_output=test_case.expected_output,
                        evaluator_name=test_case.evaluator.__name__,
                    )
                except Exception as e:
                    result = EvalResult(
                        test_id=test_case.get_test_id(),
                        passed=False,
                        score=0.0,
                        details=f"Evaluator error: {str(e)}",
                        actual_output=actual_output,
                        expected_output=test_case.expected_output,
                        evaluator_name=test_case.evaluator.__name__,
                    )

            results.add_result(result)

        self._last_results = results
        return results

    def report(
        self,
        format: Literal["terminal", "json", "csv", "markdown"] = "terminal",
    ) -> str:
        """
        Generate evaluation report.

        Args:
            format: Report format ("terminal", "json", "csv", or "markdown")

        Returns:
            Formatted report string

        Raises:
            ValueError: If no results available or unknown format
        """
        if self._last_results is None:
            raise ValueError(
                "No evaluation results available. Call run() first to generate results."
            )

        if format == "terminal":
            return ReportGenerator.generate_terminal_report(self._last_results)
        elif format == "json":
            return ReportGenerator.generate_json_report(self._last_results)
        elif format == "csv":
            return ReportGenerator.generate_csv_report(self._last_results)
        elif format == "markdown":
            return ReportGenerator.generate_markdown_report(self._last_results)
        else:
            raise ValueError(f"Unknown report format: {format}")

    def get_results(self) -> EvalResults | None:
        """
        Get the last evaluation results.

        Returns:
            EvalResults object or None if no evaluation run yet
        """
        return self._last_results

    def summary(self) -> dict:
        """
        Get summary metrics from last evaluation.

        Returns:
            Dictionary with summary statistics

        Raises:
            ValueError: If no results available
        """
        if self._last_results is None:
            raise ValueError("No evaluation results available. Call run() first.")
        return self._last_results.get_summary()
