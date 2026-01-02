"""
RAG Model Accuracy Test Script.
Tests the accuracy of the RAG system using known Q&A pairs from indexed documents.
"""

import sys
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import query_documents

@dataclass
class TestCase:
    """A single test case with question and expected answer."""
    question: str
    expected_answer: str
    expected_keywords: List[str]  # Keywords that should appear in the answer

# Test cases based on the quantum computing MCQ document
TEST_CASES = [
    TestCase(
        question="What is the architecture of a quantum computing platform primarily based on?",
        expected_answer="A",
        expected_keywords=["superposition", "entanglement"]
    ),
    TestCase(
        question="How does a qubit differ from a classical bit?",
        expected_answer="B",
        expected_keywords=["superposition", "0 and 1"]
    ),
    TestCase(
        question="What does the Bloch sphere represent?",
        expected_answer="B",
        expected_keywords=["pure states", "single qubit"]
    ),
    TestCase(
        question="What is the Hilbert space dimension for n qubits?",
        expected_answer="C",
        expected_keywords=["2^n", "2 to the power"]
    ),
    TestCase(
        question="How many basis states does a 2-qubit system have?",
        expected_answer="B",
        expected_keywords=["4", "four"]
    ),
    TestCase(
        question="What does the Pauli-X gate act like?",
        expected_answer="B",
        expected_keywords=["NOT", "bit flip"]
    ),
    TestCase(
        question="What does the Pauli-Z gate act like?",
        expected_answer="B",
        expected_keywords=["phase flip"]
    ),
    TestCase(
        question="What does the Hadamard gate transform |0⟩ into?",
        expected_answer="B",
        expected_keywords=["superposition", "|0⟩+|1⟩", "equal"]
    ),
    TestCase(
        question="What does the CNOT gate do?",
        expected_answer="B",
        expected_keywords=["flips", "control", "|1⟩"]
    ),
    TestCase(
        question="Which scientist referred to entanglement as 'spooky action at a distance'?",
        expected_answer="A",
        expected_keywords=["Einstein"]
    ),
    TestCase(
        question="What is a universal set of quantum gates?",
        expected_answer="A",
        expected_keywords=["H", "T", "CNOT"]
    ),
    TestCase(
        question="What is quantum teleportation?",
        expected_answer="D",
        expected_keywords=["entanglement", "classical", "Bell"]
    ),
    TestCase(
        question="What destroys quantum superposition?",
        expected_answer="A",
        expected_keywords=["measurement"]
    ),
    TestCase(
        question="What is the Hilbert space dimension for 3 qubits?",
        expected_answer="C",
        expected_keywords=["8"]
    ),
    TestCase(
        question="What does the Toffoli gate act on?",
        expected_answer="C",
        expected_keywords=["3 qubits", "three"]
    ),
]


def evaluate_answer(response: str, test_case: TestCase) -> Tuple[bool, str]:
    """
    Evaluate if the model's response is correct.
    
    Returns:
        Tuple of (is_correct, reason)
    """
    response_lower = response.lower()
    
    # Check if the expected answer letter is mentioned
    answer_patterns = [
        f"answer: {test_case.expected_answer.lower()}",
        f"answer is {test_case.expected_answer.lower()}",
        f"correct answer is {test_case.expected_answer.lower()}",
        f"option {test_case.expected_answer.lower()}",
        f"{test_case.expected_answer.lower()})",
        f"{test_case.expected_answer.lower()}.",
    ]
    
    answer_found = any(pattern in response_lower for pattern in answer_patterns)
    
    # Check for expected keywords
    keywords_found = sum(
        1 for keyword in test_case.expected_keywords 
        if keyword.lower() in response_lower
    )
    keyword_ratio = keywords_found / len(test_case.expected_keywords) if test_case.expected_keywords else 0
    
    # Consider correct if answer letter found OR majority of keywords present
    if answer_found:
        return True, f"Found answer {test_case.expected_answer}"
    elif keyword_ratio >= 0.5:
        return True, f"Found {keywords_found}/{len(test_case.expected_keywords)} keywords"
    else:
        return False, f"Expected {test_case.expected_answer}, keywords: {keywords_found}/{len(test_case.expected_keywords)}"


def run_accuracy_test(provider: str = None, verbose: bool = True) -> Dict:
    """
    Run accuracy test on the RAG system.
    
    Args:
        provider: LLM provider to test ('ollama', 'gemini', or None for default)
        verbose: Print detailed output
    
    Returns:
        Dictionary with test results
    """
    results = {
        "total": len(TEST_CASES),
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "accuracy": 0.0,
        "provider": provider or "default",
        "details": []
    }
    
    print(f"\n{'='*60}")
    print(f"🧪 RAG Model Accuracy Test")
    print(f"Provider: {provider or 'default'}")
    print(f"Test Cases: {len(TEST_CASES)}")
    print(f"{'='*60}\n")
    
    for i, test_case in enumerate(TEST_CASES, 1):
        try:
            if verbose:
                print(f"\n[{i}/{len(TEST_CASES)}] Testing: {test_case.question[:50]}...")
            
            start_time = time.time()
            response = query_documents(
                query=test_case.question,
                provider=provider
            )
            elapsed = time.time() - start_time
            
            answer = response.get("answer", "")
            used_provider = response.get("provider", "unknown")
            
            is_correct, reason = evaluate_answer(answer, test_case)
            
            if is_correct:
                results["correct"] += 1
                status = "✅ CORRECT"
            else:
                results["incorrect"] += 1
                status = "❌ INCORRECT"
            
            if verbose:
                print(f"   {status} ({reason}) - {elapsed:.2f}s - Provider: {used_provider}")
            
            results["details"].append({
                "question": test_case.question,
                "expected": test_case.expected_answer,
                "is_correct": is_correct,
                "reason": reason,
                "response_time": elapsed,
                "provider_used": used_provider
            })
            
        except Exception as e:
            results["errors"] += 1
            if verbose:
                print(f"   ⚠️ ERROR: {str(e)[:50]}")
            results["details"].append({
                "question": test_case.question,
                "expected": test_case.expected_answer,
                "is_correct": False,
                "reason": f"Error: {str(e)}",
                "response_time": 0,
                "provider_used": "error"
            })
    
    # Calculate final accuracy
    results["accuracy"] = (results["correct"] / results["total"]) * 100 if results["total"] > 0 else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {results['total']}")
    print(f"Correct: {results['correct']} ✅")
    print(f"Incorrect: {results['incorrect']} ❌")
    print(f"Errors: {results['errors']} ⚠️")
    print(f"\n🎯 ACCURACY: {results['accuracy']:.1f}%")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main entry point for the accuracy test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG model accuracy")
    parser.add_argument(
        "--provider", 
        choices=["ollama", "gemini"], 
        default=None,
        help="LLM provider to test"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    results = run_accuracy_test(
        provider=args.provider,
        verbose=not args.quiet
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Return exit code based on accuracy
    return 0 if results["accuracy"] >= 70 else 1


if __name__ == "__main__":
    exit(main())
