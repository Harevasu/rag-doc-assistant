"""
FastAPI Main Application.
Provides REST API endpoints for the RAG Document Assistant.
"""

import uuid
import shutil
import json
import time
import threading
from pathlib import Path
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import UPLOAD_DIR, validate_config, LLM_PROVIDER, GOOGLE_API_KEYS, OLLAMA_LLM_MODEL
from document_processor import process_document
from vector_store import add_documents, delete_document, get_all_documents, get_document_count, debug_search, hybrid_search
from rag_pipeline import query_documents, retrieve_context, format_context, format_sources, generate_response_stream
from trainer import prepare_training_data, train_model, is_model_trained, fine_tuned_model

# Indexing status tracker
class IndexingTracker:
    """Track document indexing progress and status."""
    
    def __init__(self):
        self._status: Dict[str, dict] = {}
        self._lock = threading.Lock()
    
    def start_indexing(self, document_id: str, filename: str, estimated_chunks: int = 0):
        """Mark a document as starting indexing."""
        with self._lock:
            self._status[document_id] = {
                "status": "processing",
                "filename": filename,
                "started_at": time.time(),
                "estimated_chunks": estimated_chunks,
                "chunks_processed": 0,
                "current_step": "Extracting text...",
                "error": None
            }
    
    def update_progress(self, document_id: str, chunks_processed: int, current_step: str, estimated_chunks: int = None):
        """Update indexing progress."""
        with self._lock:
            if document_id in self._status:
                self._status[document_id]["chunks_processed"] = chunks_processed
                self._status[document_id]["current_step"] = current_step
                if estimated_chunks is not None:
                    self._status[document_id]["estimated_chunks"] = estimated_chunks
    
    def complete_indexing(self, document_id: str, total_chunks: int):
        """Mark indexing as complete."""
        with self._lock:
            if document_id in self._status:
                self._status[document_id]["status"] = "completed"
                self._status[document_id]["chunks_processed"] = total_chunks
                self._status[document_id]["current_step"] = "Indexing complete!"
                self._status[document_id]["completed_at"] = time.time()
    
    def fail_indexing(self, document_id: str, error: str):
        """Mark indexing as failed."""
        with self._lock:
            if document_id in self._status:
                self._status[document_id]["status"] = "failed"
                self._status[document_id]["error"] = error
                self._status[document_id]["current_step"] = "Failed"
    
    def get_status(self, document_id: str) -> Optional[dict]:
        """Get status for a specific document."""
        with self._lock:
            status = self._status.get(document_id)
            if status:
                result = status.copy()
                # Calculate elapsed time and ETA
                if result["status"] == "processing":
                    elapsed = time.time() - result["started_at"]
                    result["elapsed_seconds"] = round(elapsed, 1)
                    
                    # Estimate remaining time based on chunks processed
                    if result["chunks_processed"] > 0 and result["estimated_chunks"] > 0:
                        rate = result["chunks_processed"] / elapsed
                        remaining = result["estimated_chunks"] - result["chunks_processed"]
                        result["eta_seconds"] = round(remaining / rate, 1) if rate > 0 else None
                    else:
                        result["eta_seconds"] = None
                return result
            return None
    
    def get_all_active(self) -> Dict[str, dict]:
        """Get all active indexing operations."""
        with self._lock:
            active = {}
            for doc_id, status in self._status.items():
                if status["status"] == "processing":
                    result = status.copy()
                    elapsed = time.time() - result["started_at"]
                    result["elapsed_seconds"] = round(elapsed, 1)
                    active[doc_id] = result
            return active
    
    def cleanup_old(self, max_age_seconds: int = 300):
        """Remove completed/failed entries older than max_age."""
        with self._lock:
            now = time.time()
            to_remove = []
            for doc_id, status in self._status.items():
                if status["status"] in ("completed", "failed"):
                    completed_at = status.get("completed_at", status["started_at"])
                    if now - completed_at > max_age_seconds:
                        to_remove.append(doc_id)
            for doc_id in to_remove:
                del self._status[doc_id]

# Global indexing tracker
indexing_tracker = IndexingTracker()

# Create FastAPI app
app = FastAPI(
    title="RAG Document Assistant",
    description="AI-powered document Q&A using Retrieval-Augmented Generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None
    provider: Optional[str] = None  # 'ollama', 'gemini', or None for default

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    chunks_used: int
    provider: Optional[str] = None  # Which provider was used

class ProviderInfo(BaseModel):
    name: str
    display_name: str
    available: bool
    model: Optional[str] = None

class ProvidersResponse(BaseModel):
    current: str
    providers: List[ProviderInfo]

class DocumentInfo(BaseModel):
    id: str
    filename: str
    total_chunks: int

class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    chunks_indexed: int
    message: str

class StatusResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int


class IndexingStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str  # "processing", "completed", "failed"
    current_step: str
    chunks_processed: int
    estimated_chunks: int
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    error: Optional[str] = None


class AllIndexingStatusResponse(BaseModel):
    indexing_documents: Dict[str, dict]
    total_active: int


class TrainRequest(BaseModel):
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5


class TrainResponse(BaseModel):
    success: bool
    message: str
    model_path: Optional[str] = None
    training_examples: Optional[int] = None


class TrainingStatusResponse(BaseModel):
    model_trained: bool
    model_path: Optional[str] = None
    provider: str


# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text", ".docx"}


@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠️ Configuration Warning: {e}")
        print("The server will start, but API calls will fail until configuration is fixed.")


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG Document Assistant"}


@app.get("/status", response_model=StatusResponse, tags=["Health"])
async def get_status():
    """Get system status and document count."""
    documents = get_all_documents()
    chunk_count = get_document_count()
    
    return StatusResponse(
        status="healthy",
        total_documents=len(documents),
        total_chunks=chunk_count
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload and index a document.
    
    Supported formats: PDF, TXT, MD, DOCX
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    # Save file to uploads directory
    file_path = UPLOAD_DIR / f"{document_id}{file_ext}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Start tracking indexing progress
        indexing_tracker.start_indexing(document_id, file.filename)
        
        # Define background processing task
        def process_and_index():
            try:
                # Update status: extracting text
                indexing_tracker.update_progress(document_id, 0, "Extracting text from document...")
                
                # Process document (extract text and chunk)
                chunks = process_document(str(file_path), document_id)
                
                # Update status: got chunks, now embedding
                indexing_tracker.update_progress(document_id, 0, "Generating embeddings...", estimated_chunks=len(chunks))
                
                # Store original filename in metadata
                for chunk in chunks:
                    chunk["metadata"]["filename"] = file.filename
                
                # Index chunks in vector store with progress updates
                indexing_tracker.update_progress(document_id, 0, f"Indexing {len(chunks)} chunks...", estimated_chunks=len(chunks))
                add_documents(chunks)
                
                # Mark as complete
                indexing_tracker.complete_indexing(document_id, len(chunks))
                print(f"✅ Successfully indexed {len(chunks)} chunks from {file.filename}")
            except Exception as e:
                error_msg = str(e)
                indexing_tracker.fail_indexing(document_id, error_msg)
                print(f"❌ Background processing error for {file.filename}: {e}")
                if file_path.exists():
                    file_path.unlink()

        # Add to background tasks
        background_tasks.add_task(process_and_index)
        
        return UploadResponse(
            success=True,
            document_id=document_id,
            filename=file.filename,
            chunks_indexed=0,  # Initial value, indexing happens in background
            message=f"Upload successful. {file.filename} is being indexed in the background."
        )
        
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error starting upload: {str(e)}")


@app.get("/providers", response_model=ProvidersResponse, tags=["Providers"])
async def get_providers():
    """Get available LLM providers and current selection."""
    providers = [
        ProviderInfo(
            name="gemini",
            display_name="🌟 Gemini API",
            available=len(GOOGLE_API_KEYS) > 0,
            model="gemini-2.0-flash"
        ),
        ProviderInfo(
            name="ollama",
            display_name="🦙 Ollama (Local)",
            available=True,  # Assume Ollama is available if configured
            model=OLLAMA_LLM_MODEL
        )
    ]
    
    return ProvidersResponse(
        current=LLM_PROVIDER,
        providers=providers
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Ask a question about the indexed documents.
    
    Uses RAG to retrieve relevant context and generate an answer.
    Optionally specify a provider to use for this query.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = query_documents(
            query=request.question,
            chat_history=request.chat_history,
            provider=request.provider
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            provider=result.get("provider")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Stream answer tokens using Server-Sent Events.
    
    Returns SSE stream with tokens and metadata.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Retrieve context first
    chunks = retrieve_context(request.question)
    context = format_context(chunks)
    sources = format_sources(chunks)
    
    async def event_generator():
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'chunks_used': len(chunks)})}\n\n"
        
        # Stream tokens
        for token in generate_response_stream(
            query=request.question,
            context=context,
            chat_history=request.chat_history,
            provider=request.provider
        ):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        
        # Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    documents = get_all_documents()
    return [DocumentInfo(**doc) for doc in documents]


@app.get("/documents/indexing", tags=["Documents"])
async def get_indexing_status():
    """
    Get status of all currently indexing documents.
    
    Returns active indexing operations with progress and ETA.
    """
    # Cleanup old entries
    indexing_tracker.cleanup_old()
    
    active = indexing_tracker.get_all_active()
    return {
        "indexing_documents": active,
        "total_active": len(active)
    }


@app.get("/documents/indexing/{document_id}", tags=["Documents"])
async def get_document_indexing_status(document_id: str):
    """
    Get indexing status for a specific document.
    
    Returns progress, current step, and ETA for the document.
    """
    status = indexing_tracker.get_status(document_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="No indexing status found for this document")
    
    return {
        "document_id": document_id,
        **status
    }


@app.delete("/documents/{document_id}", tags=["Documents"])
async def remove_document(document_id: str):
    """Remove a document from the index."""
    chunks_deleted = delete_document(document_id)
    
    if chunks_deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Also remove the file if it exists
    for ext in SUPPORTED_EXTENSIONS:
        file_path = UPLOAD_DIR / f"{document_id}{ext}"
        if file_path.exists():
            file_path.unlink()
            break
    
    return {
        "success": True,
        "message": f"Deleted document with {chunks_deleted} chunks"
    }


@app.get("/train/status", response_model=TrainingStatusResponse, tags=["Training"])
async def get_training_status():
    """Check if a trained model exists."""
    from config import DATA_DIR
    model_path = str(DATA_DIR / "trained_model") if is_model_trained() else None
    
    return TrainingStatusResponse(
        model_trained=is_model_trained(),
        model_path=model_path,
        provider=LLM_PROVIDER
    )


@app.post("/train/prepare", tags=["Training"])
async def prepare_data():
    """Prepare training data from indexed documents."""
    try:
        documents = get_all_documents()
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please upload documents first."
            )
        
        data_path = prepare_training_data()
        
        import json
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        return {
            "success": True,
            "message": f"Generated {len(data)} training examples",
            "training_examples": len(data),
            "data_path": data_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")


@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def train(request: TrainRequest = TrainRequest()):
    """
    Fine-tune the model on indexed documents.
    
    This will:
    1. Generate Q&A pairs from document chunks
    2. Fine-tune Flan-T5 on the generated data
    3. Save the model for inference
    """
    try:
        documents = get_all_documents()
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please upload documents first."
            )
        
        # Train the model
        model_path = train_model(
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate
        )
        
        return TrainResponse(
            success=True,
            message="Model trained successfully!",
            model_path=model_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


# Debug endpoints for diagnosing retrieval issues
class DebugSearchRequest(BaseModel):
    query: str
    top_k: int = 15
    use_hybrid: bool = False

class DebugSearchResult(BaseModel):
    rank: int
    id: str
    filename: str
    semantic_score: float
    keyword_score: float
    hybrid_score: float
    distance: float
    text_preview: str

class DebugSearchResponse(BaseModel):
    query: str
    total_results: int
    search_type: str
    results: List[DebugSearchResult]


@app.post("/debug/search", response_model=DebugSearchResponse, tags=["Debug"])
async def debug_search_endpoint(request: DebugSearchRequest):
    """
    Debug endpoint to diagnose retrieval issues.
    
    Shows all retrieved chunks with their scores to help identify
    why certain content may not be ranking correctly.
    """
    if request.use_hybrid:
        results = hybrid_search(request.query, top_k=request.top_k)
        search_type = "hybrid"
    else:
        results = debug_search(request.query, top_k=request.top_k)
        search_type = "semantic"
    
    formatted_results = []
    for i, r in enumerate(results, 1):
        text_preview = r.get("text_preview") or (r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"])
        formatted_results.append(DebugSearchResult(
            rank=i,
            id=r["id"],
            filename=r["metadata"].get("filename", "Unknown"),
            semantic_score=round(r.get("score", r.get("semantic_score", 0)) * 100, 2),
            keyword_score=round(r.get("keyword_score", 0) * 100, 2),
            hybrid_score=round(r.get("hybrid_score", r.get("score", 0)) * 100, 2),
            distance=round(r.get("distance", 0), 4),
            text_preview=text_preview
        ))
    
    return DebugSearchResponse(
        query=request.query,
        total_results=len(results),
        search_type=search_type,
        results=formatted_results
    )


@app.get("/debug/search", tags=["Debug"])
async def debug_search_get(query: str, top_k: int = 15, use_hybrid: bool = False):
    """GET version of debug search for easy browser testing."""
    request = DebugSearchRequest(query=query, top_k=top_k, use_hybrid=use_hybrid)
    return await debug_search_endpoint(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


