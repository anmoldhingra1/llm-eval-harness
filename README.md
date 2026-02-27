# LLM Eval Harness

[![CI](https://github.com/anmoldhingra1/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/anmoldhingra1/llm-eval-harness/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal, extensible evaluation harness for LLM applications. Define test cases, run evaluations, and get structured reports with built-in and custom evaluators.

## Overview

LLM Eval Harness provides:
- Simple test case definition (input, expected output, evaluator)
- Built-in evaluators for common scenarios (exact match, semantic similarity, JSON validity, toxicity)
- Custom evaluator support for specialized evaluation logic
- Structured JSON reports and clean terminal summaries
- Batch evaluation across multiple test cases

Perfect for evaluating chatbots, content generation systems, code-generating LLMs, and any AI application that produces text output.

## Installation

```bash
pip install llm-eval-harness
```

## Quick Start

```python
from llm_eval import EvalHarness, TestCase
from llm_eval.evaluators import exact_match, semantic_similarity

# Create harness
harness = EvalHarness(name="chatbot_eval")

# Add test cases
harness.add_test_case(
    TestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        evaluator=exact_match,
        metadata={"category": "geography"}
    )
)

harness.add_test_case(
    TestCase(
        input="Explain photosynthesis briefly.",
        expected_output="Process where plants convert sunlight to energy",
        evaluator=semantic_similarity,
        metadata={"category": "science"}
    )
)

# Run evaluation with actual model output
results = harness.run({
    "What is the capital of France?": "Paris",
    "Explain photosynthesis briefly.": "Plants use sunlight to make glucose and oxygen"
})

# Get summary
harness.report()
```

## Features

### Built-in Evaluators

- **exact_match**: Strict string equality (case-insensitive)
- **contains**: Check if output contains expected substring
- **semantic_similarity**: Compare meaning using embeddings (requires optional dependency)
- **json_valid**: Validate output is valid JSON
- **response_length**: Check output length is within bounds
- **toxicity_check**: Detect harmful content (requires optional dependency)

### Custom Evaluators

```python
from llm_eval import EvalHarness, TestCase

def custom_evaluator(actual: str, expected: str, metadata: dict) -> dict:
    """Custom evaluator must return dict with 'passed' bool and optional 'score'."""
    # Your logic here
    passed = len(actual) >= 10
    score = min(len(actual) / 100, 1.0)
    return {"passed": passed, "score": score}

harness = EvalHarness(name="custom_eval")
harness.add_test_case(
    TestCase(
        input="Your prompt",
        expected_output="Expected output",
        evaluator=custom_evaluator
    )
)
```

### Batch Evaluation

```python
# Run evaluation across multiple outputs
results = harness.run({
    "input_1": "output_1",
    "input_2": "output_2",
    "input_3": "output_3",
})

# Get detailed reports
json_report = harness.report(format="json")
terminal_summary = harness.report(format="terminal")
```

## API Reference

### EvalHarness

Main harness class for managing evaluations.

#### `__init__(name: str, description: str = "")`
Initialize evaluation harness.

#### `add_test_case(test_case: TestCase) -> None`
Add a test case to the harness.

#### `run(outputs: dict[str, str]) -> EvalResults`
Run evaluation with model outputs.

**Args:**
- `outputs`: Dict mapping inputs to model outputs

**Returns:**
- `EvalResults` object with detailed results

#### `report(format: str = "terminal") -> str`
Generate evaluation report.

**Args:**
- `format`: "terminal" for human-readable output, "json" for structured data

**Returns:**
- Formatted report string

### TestCase

Dataclass representing a single test case.

```python
@dataclass
class TestCase:
    input: str                              # Input to the model
    expected_output: str                    # Expected/reference output
    evaluator: Callable[[str, str, dict], dict]  # Evaluation function
    metadata: dict = field(default_factory=dict)  # Optional metadata
```

### Evaluators

All evaluators have signature:
```python
def evaluator(actual: str, expected: str, metadata: dict) -> dict:
    """
    Returns:
        {"passed": bool, "score": float (optional), "details": str (optional)}
    """
```

## Use Cases

### Chatbot Quality Assurance
```python
from llm_eval import EvalHarness, TestCase
from llm_eval.evaluators import semantic_similarity

harness = EvalHarness(name="chatbot_qa")

test_cases = [
    TestCase(
        input="How do I reset my password?",
        expected_output="Go to login, click forgot password, check email",
        evaluator=semantic_similarity,
        metadata={"category": "support"}
    ),
    # ... more test cases
]

for tc in test_cases:
    harness.add_test_case(tc)

# Evaluate with your chatbot
results = harness.run(chatbot_outputs)
harness.report()
```

### Content Generation
```python
from llm_eval.evaluators import response_length, contains

harness = EvalHarness(name="article_generation")

harness.add_test_case(
    TestCase(
        input="Write a 500-word article about AI",
        expected_output="Article content",
        evaluator=response_length,
        metadata={"min_length": 400, "max_length": 600}
    )
)
```

### Code Generation
```python
from llm_eval.evaluators import json_valid

harness = EvalHarness(name="code_gen")

harness.add_test_case(
    TestCase(
        input="Generate a JSON schema for a user",
        expected_output='{"type": "object", ...}',
        evaluator=json_valid
    )
)
```

## Architecture

```
llm-eval-harness/
├── llm_eval/
│   ├── harness.py       # Main EvalHarness class
│   ├── evaluators.py    # Built-in evaluators
│   ├── report.py        # Report generation
│   ├── test_case.py     # TestCase dataclass
│   └── __init__.py      # Public API
├── examples/
│   └── evaluate_chatbot.py
└── tests/
```

## Development

### Setting Up for Development

Clone and install with dev dependencies:
```bash
git clone https://github.com/anmoldhingra1/llm-eval-harness.git
cd llm-eval-harness
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/ -v
```

### Linting

```bash
ruff check . --fix
black .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## License

MIT License - see LICENSE file for details.

---

Built by [Anmol Dhingra](https://anmol.one)
