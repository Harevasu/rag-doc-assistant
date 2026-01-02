"""
Embeddings Module.
Handles generation of text embeddings using Ollama (local) or Gemini (API).
Optimized for performance with parallel processing.
"""

from typing import List
import concurrent.futures
from config import (
    LLM_PROVIDER, 
    OLLAMA_HOST, 
    OLLAMA_EMBED_MODEL,
    GOOGLE_API_KEYS, 
    EMBEDDING_MODEL
)

# Initialize Ollama client once to reuse connections
_ollama_client = None

def get_ollama_client():
    """Get or initialize the shared Ollama client."""
    global _ollama_client
    if _ollama_client is None:
        import ollama
        _ollama_client = ollama.Client(host=OLLAMA_HOST)
    return _ollama_client


def get_embedding(text: str, task_type: str = "retrieval_query") -> List[float]:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text to embed
        task_type: For Gemini - "retrieval_document" for docs, "retrieval_query" for queries
        
    Returns:
        Embedding vector as a list of floats
    """
    # Clean and truncate text if needed
    text = text.replace("\n", " ").strip()
    if not text:
        text = " "  # Handle empty text
    
    if LLM_PROVIDER == "ollama":
        return _get_ollama_embedding(text)
    else:
        return _get_gemini_embedding(text, task_type=task_type)


def get_embeddings(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """
    Generate embeddings for multiple texts (typically documents).
    Uses native batching for Gemini and parallel requests for Ollama.
    
    Args:
        texts: List of texts to embed
        task_type: For Gemini - "retrieval_document" for docs, "retrieval_query" for queries
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    # Clean texts
    cleaned_texts = []
    for text in texts:
        cleaned = text.replace("\n", " ").strip()
        cleaned_texts.append(cleaned if cleaned else " ")

    if LLM_PROVIDER == "gemini":
        return _get_gemini_embeddings_batch(cleaned_texts, task_type=task_type)
    
    # For Ollama, continue using parallel processing
    max_workers = min(10, len(cleaned_texts)) # Dynamic workers
    
    embeddings = [None] * len(cleaned_texts)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(get_embedding, text, task_type): i 
            for i, text in enumerate(cleaned_texts)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                embeddings[index] = future.result()
            except Exception as e:
                print(f"❌ Error generating embedding for chunk {index}: {e}")
                embeddings[index] = [0.0] * 768  # Fallback
    
    return embeddings


def _get_gemini_embeddings_batch(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Generate embeddings for a batch of texts using Gemini API."""
    import google.generativeai as genai
    
    if not GOOGLE_API_KEYS:
        raise ValueError("No Gemini API keys configured")
    
    # Configure with the first available key
    genai.configure(api_key=GOOGLE_API_KEYS[0])
    
    # Gemini supports batching via embed_content with a list of strings
    # Note: There might be a limit on batch size (usually ~100)
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type=task_type
            )
            # For batch input, result['embedding'] is a list of embeddings
            embeddings_result = result['embedding']
            if isinstance(embeddings_result[0], list):
                # Batch result: list of embedding vectors
                all_embeddings.extend(embeddings_result)
            else:
                # Single result wrapped in list: just the embedding vector
                all_embeddings.append(embeddings_result)
        except Exception as e:
            print(f"❌ Error in Gemini batch embedding: {e}")
            # Fallback to individual embeddings or zero vectors for this batch
            for _ in batch:
                all_embeddings.append([0.0] * 768)
                
    return all_embeddings


def _get_ollama_embedding(text: str) -> List[float]:
    """Generate embedding using shared Ollama client."""
    client = get_ollama_client()
    response = client.embeddings(
        model=OLLAMA_EMBED_MODEL,
        prompt=text
    )
    return response['embedding']


def _get_gemini_embedding(text: str, task_type: str = "retrieval_query") -> List[float]:
    """Generate embedding using Gemini API."""
    import google.generativeai as genai
    
    if not GOOGLE_API_KEYS:
        raise ValueError("No Gemini API keys configured")
    
    # Simple rotation or just use the first key
    genai.configure(api_key=GOOGLE_API_KEYS[0])
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type
    )
    return result['embedding']
