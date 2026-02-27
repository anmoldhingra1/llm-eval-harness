"""LLM Eval Harness - Minimal evaluation framework for LLM applications."""

from llm_eval.harness import EvalHarness
from llm_eval.test_case import TestCase, EvalResult, EvalResults
from llm_eval import evaluators

__version__ = "0.1.0"
__author__ = "Anmol Dhingra"

__all__ = [
    "EvalHarness",
    "TestCase",
    "EvalResult",
    "EvalResults",
    "evaluators",
]
