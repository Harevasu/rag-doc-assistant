
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from rag_pipeline import generate_response

class TestLLMFallback(unittest.TestCase):

    def setUp(self):
        self.query = "What is the capital of France?"
        self.context = "France is a country in Europe. Its capital is Paris."

    @patch("rag_pipeline._generate_gemini_response")
    @patch("rag_pipeline._generate_ollama_response")
    def test_gemini_to_ollama_fallback(self, mock_ollama, mock_gemini):
        # Mock Gemini to fail with a quota error
        mock_gemini.side_effect = Exception("429: Quota exceeded")
        mock_ollama.return_value = "Paris (from Ollama)"

        answer, provider = generate_response(self.query, self.context, provider="gemini")

        self.assertEqual(answer, "Paris (from Ollama)")
        self.assertEqual(provider, "ollama")
        mock_gemini.assert_called_once()
        mock_ollama.assert_called_once()

    @patch("rag_pipeline._generate_gemini_response")
    @patch("rag_pipeline._generate_ollama_response")
    def test_ollama_to_gemini_fallback(self, mock_ollama, mock_gemini):
        # Mock Ollama to fail with a CUDA error
        mock_ollama.side_effect = Exception("CUDA out of memory")
        mock_gemini.return_value = "Paris (from Gemini)"

        answer, provider = generate_response(self.query, self.context, provider="ollama")

        self.assertEqual(answer, "Paris (from Gemini)")
        self.assertEqual(provider, "gemini")
        mock_ollama.assert_called_once()
        mock_gemini.assert_called_once()

    @patch("rag_pipeline._generate_gemini_response")
    def test_gemini_success(self, mock_gemini):
        mock_gemini.return_value = "Paris (from Gemini)"

        answer, provider = generate_response(self.query, self.context, provider="gemini")

        self.assertEqual(answer, "Paris (from Gemini)")
        self.assertEqual(provider, "gemini")
        mock_gemini.assert_called_once()

if __name__ == "__main__":
    unittest.main()
