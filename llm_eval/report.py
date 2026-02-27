"""Report generation for evaluation results."""

import json

from llm_eval.test_case import EvalResults


class ReportGenerator:
    """Generate evaluation reports in various formats."""

    @staticmethod
    def generate_terminal_report(results: EvalResults) -> str:
        """
        Generate a human-readable terminal report.

        Args:
            results: EvalResults object to report on

        Returns:
            Formatted report string
        """
        summary = results.get_summary()

        lines = [
            "",
            "=" * 70,
            f"Evaluation Report: {results.harness_name}",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 70,
            f"Total Tests:    {summary['total']}",
            f"Passed:         {summary['passed']}",
            f"Failed:         {summary['failed']}",
            f"Pass Rate:      {summary['pass_rate']:.1%}",
            f"Avg Score:      {summary['average_score']:.2f}",
            "",
            "DETAILS",
            "-" * 70,
        ]

        for i, result in enumerate(results.results, 1):
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"\n{i}. {result.test_id} [{status}]")
            lines.append(f"   Evaluator: {result.evaluator_name}")
            if result.score < 1.0:
                lines.append(f"   Score: {result.score:.2f}")
            if result.details:
                lines.append(f"   {result.details}")

        lines.extend(["", "=" * 70, ""])

        return "\n".join(lines)

    @staticmethod
    def generate_json_report(results: EvalResults) -> str:
        """
        Generate a JSON report.

        Args:
            results: EvalResults object to report on

        Returns:
            JSON formatted report string
        """
        data = results.to_dict()
        return json.dumps(data, indent=2)

    @staticmethod
    def generate_csv_report(results: EvalResults) -> str:
        """
        Generate a CSV report.

        Args:
            results: EvalResults object to report on

        Returns:
            CSV formatted report string
        """
        lines = ["test_id,passed,score,evaluator,details"]

        for result in results.results:
            # Escape quotes in details
            details = result.details.replace('"', '""')
            line = f'{result.test_id},{result.passed},{result.score:.2f},{result.evaluator_name},"{details}"'
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def generate_markdown_report(results: EvalResults) -> str:
        """
        Generate a Markdown report.

        Args:
            results: EvalResults object to report on

        Returns:
            Markdown formatted report string
        """
        summary = results.get_summary()

        lines = [
            f"# Evaluation Report: {results.harness_name}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total Tests | {summary['total']} |",
            f"| Passed | {summary['passed']} |",
            f"| Failed | {summary['failed']} |",
            f"| Pass Rate | {summary['pass_rate']:.1%} |",
            f"| Avg Score | {summary['average_score']:.2f} |",
            "",
            "## Results",
            "",
        ]

        for i, result in enumerate(results.results, 1):
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(f"### {i}. {result.test_id} - {status}")
            lines.append("")
            lines.append(f"**Evaluator:** {result.evaluator_name}")
            if result.score < 1.0:
                lines.append(f"**Score:** {result.score:.2f}")
            if result.details:
                lines.append(f"**Details:** {result.details}")
            lines.append("")

        return "\n".join(lines)
