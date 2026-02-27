"""Pytest configuration and shared fixtures."""

import pytest
from llm_eval import evaluators
from llm_eval.test_case import TestCase


@pytest.fixture
def sample_test_case():
    """Create a sample test case for testing."""
    return TestCase(
        input="What is 2+2?",
        expected_output="4",
        evaluator=evaluators.exact_match,
        metadata={"category": "math"}
    )


@pytest.fixture
def sample_test_cases():
    """Create multiple sample test cases."""
    return [
        TestCase(
            input="What is the capital of France?",
            expected_output="Paris",
            evaluator=evaluators.exact_match,
        ),
        TestCase(
            input="Explain photosynthesis",
            expected_output="Plants convert sunlight to energy",
            evaluator=evaluators.semantic_similarity,
        ),
        TestCase(
            input='Generate {"key": "value"}',
            expected_output='{"test": "json"}',
            evaluator=evaluators.json_valid,
        ),
    ]


@pytest.fixture
def sample_outputs():
    """Create sample model outputs for testing."""
    return {
        "What is the capital of France?": "Paris",
        "Explain photosynthesis": "Plants use sunlight to make glucose",
        'Generate {"key": "value"}': '{"test": "json"}',
    }
