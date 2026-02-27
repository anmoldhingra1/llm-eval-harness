"""Test case definition for evaluation harness."""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TestCase:
    """
    Represents a single test case for LLM evaluation.

    Attributes:
        input: The input prompt or query
        expected_output: The expected or reference output
        evaluator: Callable that compares actual vs expected output
        metadata: Optional metadata for the test case
    """

    input: str
    expected_output: str
    evaluator: Callable[[str, str, dict], dict]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate test case fields."""
        if not self.input:
            raise ValueError("input cannot be empty")
        if not self.expected_output:
            raise ValueError("expected_output cannot be empty")
        if not callable(self.evaluator):
            raise ValueError("evaluator must be callable")

    def get_test_id(self) -> str:
        """
        Generate a unique identifier for this test case.

        Returns:
            String ID based on input hash
        """
        return f"test_{hash(self.input) % 10000}"

    def to_dict(self) -> dict:
        """
        Convert test case to dictionary (excluding evaluator function).

        Returns:
            Dictionary representation
        """
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "evaluator": self.evaluator.__name__,
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    """
    Result of evaluating a single test case.

    Attributes:
        test_id: Identifier for the test case
        passed: Whether the test passed
        score: Numeric score (0.0-1.0), if applicable
        details: Optional details about the result
        actual_output: The actual output from the model
        expected_output: The expected output
        evaluator_name: Name of the evaluator used
    """

    test_id: str
    passed: bool
    score: float = 0.0
    details: str = ""
    actual_output: str = ""
    expected_output: str = ""
    evaluator_name: str = ""

    def __post_init__(self) -> None:
        """Set default score based on passed status if not already set."""
        if self.score == 0.0 and self.passed:
            self.score = 1.0

    def to_dict(self) -> dict:
        """
        Convert result to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "test_id": self.test_id,
            "passed": self.passed,
            "score": self.score,
            "details": self.details,
            "evaluator": self.evaluator_name,
        }


@dataclass
class EvalResults:
    """
    Aggregated results from running evaluations.

    Attributes:
        harness_name: Name of the harness
        results: List of individual test results
    """

    harness_name: str
    results: list[EvalResult] = field(default_factory=list)

    def add_result(self, result: EvalResult) -> None:
        """
        Add a result to the collection.

        Args:
            result: EvalResult to add
        """
        self.results.append(result)

    def get_summary(self) -> dict:
        """
        Get summary statistics.

        Returns:
            Dictionary with summary metrics
        """
        if not self.results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "average_score": 0.0,
            }

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        avg_score = sum(r.score for r in self.results) / len(self.results)

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results),
            "average_score": avg_score,
        }

    def to_dict(self) -> dict:
        """
        Convert all results to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "harness_name": self.harness_name,
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self.results],
        }
