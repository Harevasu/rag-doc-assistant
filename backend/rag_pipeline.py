"""
RAG Pipeline Module.
Orchestrates retrieval and LLM response generation using Ollama (local) or Gemini (API).
"""

from typing import List, Dict, Any
import time

from config import (
    LLM_PROVIDER,
    OLLAMA_HOST,
    OLLAMA_LLM_MODEL,
    GOOGLE_API_KEYS,
    LLM_MODEL,
    TOP_K_RESULTS
)
from vector_store import search


SYSTEM_PROMPT = """You are a helpful document assistant. Your role is to answer questions based on the provided document context.

Guidelines:
1. Answer questions accurately based ONLY on the provided context
2. If the context doesn't contain enough information to answer, say so clearly
3. Cite specific parts of the documents when relevant
4. Be concise but thorough
5. If asked about something not in the documents, clarify that you can only answer based on the uploaded documents

IMPORTANT - For Multiple Choice Questions (MCQs):
- If the document contains an MCQ with options and an "ANSWER:" line, USE THAT EXACT ANSWER
- Do NOT make up your own answer - extract the answer directly from the document
- Quote the answer as it appears in the source document
- Example: If you see "ANSWER: B" in the document, report that B is the correct answer

Always maintain a helpful, professional tone."""


def retrieve_context(query: str, top_k: int = TOP_K_RESULTS, min_score: float = 0.5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant document chunks for a query.
    
    Args:
        query: User's question
        top_k: Number of chunks to retrieve
        min_score: Minimum similarity score (0-1) to include a chunk
        
    Returns:
        List of relevant chunks with metadata
    """
    results = search(query, top_k=top_k)
    
    # Filter out low-relevance chunks to avoid confusing the LLM
    filtered_results = [r for r in results if r.get("score", 0) >= min_score]
    
    # If no results meet the threshold, return the best one anyway
    if not filtered_results and results:
        filtered_results = [results[0]]
    
    return filtered_results


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant documents found."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk["metadata"].get("filename", "Unknown")
        chunk_idx = chunk["metadata"].get("chunk_index", 0) + 1
        text = chunk["text"]
        
        context_parts.append(
            f"[Source {i}: {filename} (Section {chunk_idx})]\n{text}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def format_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format source information for citation.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of source citation dictionaries
    """
    sources = []
    seen_docs = set()
    
    for chunk in chunks:
        doc_id = chunk["metadata"].get("document_id")
        filename = chunk["metadata"].get("filename", "Unknown")
        
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "document_id": doc_id,
                "filename": filename,
                "relevance_score": round(chunk.get("score", 0) * 100, 1)
            })
    
    return sources


def generate_response(
    query: str,
    context: str,
    chat_history: List[Dict[str, str]] = None,
    provider: str = None
) -> str:
    """
    Generate a response using LLM with retrieved context.
    
    Args:
        query: User's question
        context: Formatted document context
        chat_history: Optional list of previous messages
        provider: Optional provider override ('ollama', 'gemini', 'finetuned')
        
    Returns:
        Tuple of (generated response text, actual provider used)
    """
    # Use specified provider or fall back to config default
    active_provider = provider.lower() if provider else LLM_PROVIDER
    
    # Build the prompt
    prompt = f"""{SYSTEM_PROMPT}

Based on the following document excerpts, please answer the question.

DOCUMENT CONTEXT:
{context}

QUESTION: {query}

Please provide a helpful, accurate answer based on the context above. If the context doesn't contain relevant information, say so."""

    # Add chat history context if provided
    if chat_history:
        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history[-4:]
        ])
        prompt = f"Previous conversation:\n{history_text}\n\n{prompt}"

    # Try primary provider with fallback
    try:
        if active_provider == "finetuned":
            return _generate_finetuned_response(query, context), "finetuned"
        elif active_provider == "ollama":
            try:
                return _generate_ollama_response(prompt), "ollama"
            except Exception as e:
                error_str = str(e).lower()
                # Check for GPU/memory errors - fallback to Gemini
                if "cuda" in error_str or "memory" in error_str or "allocate" in error_str:
                    print(f"⚠️ Ollama error ({e}), falling back to Gemini...")
                    return _generate_gemini_response(prompt), "gemini"
                raise
        else:
            try:
                return _generate_gemini_response(prompt), "gemini"
            except Exception as e:
                error_str = str(e).lower()
                # Check for Gemini quota/rate limit errors - fallback to Ollama
                if "429" in error_str or "quota" in error_str:
                    print(f"⚠️ Gemini error ({e}), falling back to Ollama...")
                    return _generate_ollama_response(prompt), "ollama"
                raise
    except Exception as e:
        raise Exception(f"Generation failed for provider {active_provider}: {str(e)}")


def _generate_finetuned_response(query: str, context: str) -> str:
    """Generate response using the fine-tuned model."""
    from trainer import fine_tuned_model
    return fine_tuned_model.generate(query, context)


def _generate_ollama_response(prompt: str) -> str:
    """Generate response using Ollama."""
    import ollama
    
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


def _generate_gemini_response(prompt: str, max_retries: int = None) -> str:
    """Generate response using Gemini API with key rotation."""
    import google.generativeai as genai
    
    if not GOOGLE_API_KEYS:
        raise ValueError("No Gemini API keys configured")
    
    if max_retries is None:
        max_retries = len(GOOGLE_API_KEYS)
    
    last_error = None
    for i in range(max_retries):
        try:
            api_key = GOOGLE_API_KEYS[i % len(GOOGLE_API_KEYS)]
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(LLM_MODEL)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1024
                )
            )
            return response.text
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                print(f"⚠️ Rate limit hit on key {i + 1}, trying next...")
                time.sleep(1)
                continue
            raise
    
    raise last_error


def query_documents(
    query: str,
    chat_history: List[Dict[str, str]] = None,
    provider: str = None
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve context and generate response.
    
    Args:
        query: User's question
        chat_history: Optional conversation history
        provider: Optional LLM provider ('ollama', 'gemini', 'finetuned')
        
    Returns:
        Dictionary with answer, sources, and provider used
    """
    # Retrieve relevant chunks
    chunks = retrieve_context(query)
    
    # Format context for LLM
    context = format_context(chunks)
    
    # Generate response with specified provider
    answer, used_provider = generate_response(query, context, chat_history, provider)
    
    # Format sources
    sources = format_sources(chunks)
    
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "provider": used_provider
    }

