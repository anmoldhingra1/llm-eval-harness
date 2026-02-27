# Contributing to LLM Eval Harness

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.9+
- Git

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/anmoldhingra1/llm-eval-harness.git
cd llm-eval-harness
```

2. Install the package in development mode:
```bash
pip install -e .[dev]
```

3. Verify the setup by running tests:
```bash
pytest tests/ -v
```

## Development Workflow

### Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- Line length: 100 characters (enforced by Black and Ruff)
- Use [Black](https://github.com/psf/black) for code formatting
- Use [Ruff](https://github.com/astral-sh/ruff) for linting

Format your code before submitting:
```bash
black .
ruff check . --fix
```

### Testing

Write tests for new features and bug fixes. Tests should be:

- Located in `tests/` directory
- Named with `test_` prefix
- Using `pytest` framework
- Comprehensive and cover edge cases

Run tests locally:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=llm_eval --cov-report=html
```

### Type Hints

We encourage (but don't require) type hints for public APIs. Use tools like `mypy` to check:
```bash
mypy llm_eval/
```

## Adding Evaluators

To add a new built-in evaluator:

1. Add the evaluator function to `llm_eval/evaluators.py`
2. Follow the signature: `def evaluator(actual: str, expected: str, metadata: dict) -> dict`
3. Return a dict with at minimum: `{"passed": bool}`
4. Optionally include: `{"score": float, "details": str}`
5. Add comprehensive tests in `tests/test_evaluators.py`
6. Update the README with documentation

Example:
```python
def my_evaluator(actual: str, expected: str, metadata: dict) -> dict:
    """Evaluate something specific."""
    passed = actual.lower() == expected.lower()
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": f"Comparison result: {passed}"
    }
```

## Reporting Issues

When reporting issues:

1. Check existing issues to avoid duplicates
2. Provide a minimal reproducible example
3. Include Python version and OS information
4. Describe expected vs actual behavior

## Submitting Changes

1. Create a descriptive branch name: `feature/my-feature` or `fix/my-bug`
2. Make focused commits with clear messages
3. Write or update tests as needed
4. Ensure all tests pass: `pytest tests/ -v`
5. Ensure linting passes: `ruff check . && black --check .`
6. Submit a pull request with a clear description

## Code Review

- Be respectful and constructive
- Respond to feedback promptly
- Push new commits to address feedback rather than force-pushing

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or discussion for questions about contributing.
