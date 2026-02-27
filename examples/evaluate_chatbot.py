"""Example: Evaluate a chatbot across multiple test cases."""

from llm_eval import EvalHarness, TestCase
from llm_eval.evaluators import (
    semantic_similarity,
    exact_match,
    contains,
    response_length,
)


def example_basic_evaluation() -> None:
    """Basic chatbot evaluation example."""
    print("=" * 70)
    print("Example 1: Basic Chatbot Evaluation")
    print("=" * 70)

    # Create evaluation harness
    harness = EvalHarness(
        name="chatbot_quality",
        description="Evaluate chatbot response quality",
    )

    # Add test cases
    test_cases = [
        TestCase(
            input="What is the capital of France?",
            expected_output="Paris",
            evaluator=exact_match,
            metadata={"category": "geography"},
        ),
        TestCase(
            input="Tell me about photosynthesis.",
            expected_output="Process where plants use sunlight to produce energy",
            evaluator=semantic_similarity,
            metadata={"category": "biology", "min_similarity": 0.5},
        ),
        TestCase(
            input="How do I reset my password?",
            expected_output="Go to login page, click forgot password, check email",
            evaluator=semantic_similarity,
            metadata={"category": "support"},
        ),
        TestCase(
            input="Write a short poem about nature.",
            expected_output="A poem about nature",
            evaluator=response_length,
            metadata={"min_length": 50, "max_length": 300},
        ),
    ]

    harness.add_test_cases(test_cases)

    # Simulate chatbot outputs
    chatbot_outputs = {
        "What is the capital of France?": "Paris",
        "Tell me about photosynthesis.": "Photosynthesis is the process where plants convert sunlight into chemical energy through glucose production.",
        "How do I reset my password?": "You can reset your password by going to the login page, clicking the forgot password link, and following the instructions sent to your email.",
        "Write a short poem about nature.": "Green leaves dance in morning light,\nBirds sing songs of pure delight,\nRiver flows through meadows wide,\nNature's beauty, our guide.",
    }

    # Run evaluation
    harness.run(chatbot_outputs)

    # Print terminal report
    print(harness.report(format="terminal"))

    # Print summary
    summary = harness.summary()
    print("\nQuick Summary:")
    print(f"  Pass Rate: {summary['pass_rate']:.1%}")
    print(f"  Average Score: {summary['average_score']:.2f}")


def example_custom_evaluator() -> None:
    """Example with custom evaluator function."""
    print("\n" + "=" * 70)
    print("Example 2: Using Custom Evaluator")
    print("=" * 70)

    def sentiment_positivity(actual: str, expected: str, metadata: dict) -> dict:
        """Custom evaluator for sentiment positivity."""
        positive_words = {"great", "excellent", "amazing", "wonderful", "fantastic"}
        actual_lower = actual.lower()

        found_positive = any(word in actual_lower for word in positive_words)
        word_count = len(actual.split())

        return {
            "passed": found_positive,
            "score": 0.8 if found_positive else 0.2,
            "details": f"{'Positive sentiment detected' if found_positive else 'No positive sentiment detected'} ({word_count} words)",
        }

    harness = EvalHarness(name="sentiment_eval")

    harness.add_test_case(
        TestCase(
            input="How is your experience with our product?",
            expected_output="Positive feedback",
            evaluator=sentiment_positivity,
        )
    )

    outputs = {
        "How is your experience with our product?": "It's been absolutely fantastic! The product is excellent and works wonderfully.",
    }

    harness.run(outputs)
    print(harness.report(format="terminal"))


def example_multiple_formats() -> None:
    """Example showing different report formats."""
    print("\n" + "=" * 70)
    print("Example 3: Multiple Report Formats")
    print("=" * 70)

    harness = EvalHarness(name="multi_format_eval")

    harness.add_test_case(
        TestCase(
            input="What is 2+2?",
            expected_output="4",
            evaluator=exact_match,
        )
    )

    harness.add_test_case(
        TestCase(
            input="Say hello",
            expected_output="Hello there!",
            evaluator=contains,
        )
    )

    outputs = {
        "What is 2+2?": "The answer is 4",
        "Say hello": "Hello! How can I help you?",
    }

    harness.run(outputs)

    # Terminal format
    print("\n--- TERMINAL FORMAT ---")
    print(harness.report(format="terminal"))

    # JSON format
    print("\n--- JSON FORMAT (First 200 chars) ---")
    json_report = harness.report(format="json")
    print(json_report[:200] + "...")

    # CSV format
    print("\n--- CSV FORMAT ---")
    csv_report = harness.report(format="csv")
    print(csv_report)

    # Markdown format (first part)
    print("\n--- MARKDOWN FORMAT (First 300 chars) ---")
    md_report = harness.report(format="markdown")
    print(md_report[:300] + "...")


def example_batch_evaluation() -> None:
    """Example of batch evaluation across many test cases."""
    print("\n" + "=" * 70)
    print("Example 4: Batch Evaluation")
    print("=" * 70)

    harness = EvalHarness(name="batch_qa")

    # Create many test cases
    qa_pairs = [
        ("What is AI?", "AI is artificial intelligence", semantic_similarity),
        ("Who invented the internet?", "ARPANET pioneers", semantic_similarity),
        ("What year did WWII end?", "1945", exact_match),
        ("Name a planet", "Jupiter", contains),
        ("Define photosynthesis", "Process where plants convert sunlight", semantic_similarity),
    ]

    for input_text, expected, evaluator in qa_pairs:
        harness.add_test_case(
            TestCase(
                input=input_text,
                expected_output=expected,
                evaluator=evaluator,
                metadata={"batch": "qa_round_1"},
            )
        )

    # Simulate model outputs
    model_outputs = {
        "What is AI?": "Artificial Intelligence refers to computer systems designed to perform tasks that typically require human intelligence.",
        "Who invented the internet?": "The internet was developed from ARPANET, created by pioneers like Vint Cerf and Bob Kahn.",
        "What year did WWII end?": "World War II ended in 1945",
        "Name a planet": "Jupiter is the largest planet in our solar system",
        "Define photosynthesis": "Photosynthesis is a process used by plants to convert light energy into chemical energy for growth",
    }

    harness.run(model_outputs)
    print(harness.report(format="terminal"))


if __name__ == "__main__":
    example_basic_evaluation()
    example_custom_evaluator()
    example_multiple_formats()
    example_batch_evaluation()
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
