"""
Vector Store Module.
Handles ChromaDB operations for storing and retrieving document embeddings.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

from config import DATA_DIR, COLLECTION_NAME, TOP_K_RESULTS
from embeddings import get_embeddings, get_embedding

# Global variables for lazy initialization
_chroma_client = None
_collection = None

def get_collection():
    """Lazy initialization of ChromaDB client and collection."""
    global _chroma_client, _collection
    if _collection is None:
        try:
            print(f"📦 Initializing ChromaDB at {DATA_DIR / 'chroma'}...")
            _chroma_client = chromadb.PersistentClient(
                path=str(DATA_DIR / "chroma"),
                settings=Settings(anonymized_telemetry=False)
            )
            _collection = _chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ ChromaDB collection initialized.")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            raise e
    return _collection


def add_documents(chunks: List[Dict[str, Any]]) -> int:
    """
    Add document chunks to the vector store.
    """
    if not chunks:
        return 0
    
    collection = get_collection()
    
    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Generate embeddings for all chunks
    embeddings = get_embeddings(texts)
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    return len(chunks)


def search(
    query: str,
    top_k: int = TOP_K_RESULTS,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Perform semantic search for relevant document chunks.
    """
    collection = get_collection()
    
    # Generate query embedding
    query_embedding = get_embedding(query)
    
    # Search in collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filter_metadata,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted_results = []
    if results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB cosine distance ranges from 0 (identical) to 2 (opposite)
            # Convert to similarity score: 1 - (distance / 2) gives 0.5 to 1 range
            # For better UX, map cosine distance directly: score = 1 - (distance / 2)
            distance = results["distances"][0][i]
            # Cosine distance: 0 = identical (100%), 1 = orthogonal (50%), 2 = opposite (0%)
            score = max(0, 1 - (distance / 2))
            
            formatted_results.append({
                "id": chunk_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "score": score
            })
    
    return formatted_results


def _compute_keyword_score(query: str, text: str) -> float:
    """
    Compute a simple keyword matching score (BM25-style).
    Returns a score between 0 and 1.
    """
    import re
    
    # Tokenize and normalize
    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
    text_lower = text.lower()
    
    if not query_terms:
        return 0.0
    
    # Count matching terms
    matches = sum(1 for term in query_terms if term in text_lower)
    
    # Bonus for exact phrase match
    phrase_bonus = 0.2 if query.lower() in text_lower else 0.0
    
    # Score = proportion of query terms found + phrase bonus
    base_score = matches / len(query_terms)
    
    return min(1.0, base_score + phrase_bonus)


def hybrid_search(
    query: str,
    top_k: int = TOP_K_RESULTS,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic and keyword matching.
    
    Args:
        query: Search query
        top_k: Number of results to return
        semantic_weight: Weight for semantic similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)
        filter_metadata: Optional metadata filter
        
    Returns:
        List of results ranked by combined score
    """
    # Get more results initially to re-rank
    semantic_results = search(query, top_k=top_k * 2, filter_metadata=filter_metadata)
    
    if not semantic_results:
        return []
    
    # Compute hybrid scores
    for result in semantic_results:
        semantic_score = result.get("score", 0)
        keyword_score = _compute_keyword_score(query, result["text"])
        
        # Combined score
        result["semantic_score"] = semantic_score
        result["keyword_score"] = keyword_score
        result["hybrid_score"] = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
    
    # Sort by hybrid score and return top-k
    semantic_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    return semantic_results[:top_k]


def debug_search(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """
    Debug search that returns extended information about all retrieved chunks.
    Shows both semantic and keyword scores for diagnosis.
    """
    results = search(query, top_k=top_k)
    
    for result in results:
        keyword_score = _compute_keyword_score(query, result["text"])
        result["keyword_score"] = keyword_score
        result["text_preview"] = result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"]
    
    return results


def delete_document(document_id: str) -> int:
    """
    Delete all chunks associated with a document.
    """
    collection = get_collection()
    
    # Get all chunks for this document
    results = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )
    
    if not results["ids"]:
        return 0
    
    # Delete chunks
    collection.delete(ids=results["ids"])
    
    return len(results["ids"])


def get_all_documents() -> List[Dict[str, Any]]:
    """
    Get list of all indexed documents (unique by document_id).
    """
    collection = get_collection()
    
    # Get all items from collection
    results = collection.get(include=["metadatas"])
    
    if not results["ids"]:
        return []
    
    # Extract unique documents
    documents = {}
    for metadata in results["metadatas"]:
        doc_id = metadata.get("document_id")
        if doc_id and doc_id not in documents:
            documents[doc_id] = {
                "id": doc_id,
                "filename": metadata.get("filename", "Unknown"),
                "total_chunks": metadata.get("total_chunks", 0)
            }
    
    return list(documents.values())


def get_document_count() -> int:
    """Get total number of chunks in the vector store."""
    try:
        collection = get_collection()
        return collection.count()
    except Exception:
        return 0
