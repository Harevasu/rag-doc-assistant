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
            formatted_results.append({
                "id": chunk_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
    
    return formatted_results


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
