"""
FastAPI Main Application.
Provides REST API endpoints for the RAG Document Assistant.
"""

import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import UPLOAD_DIR, validate_config, LLM_PROVIDER, GOOGLE_API_KEYS, OLLAMA_LLM_MODEL
from document_processor import process_document
from vector_store import add_documents, delete_document, get_all_documents, get_document_count
from rag_pipeline import query_documents
from trainer import prepare_training_data, train_model, is_model_trained, fine_tuned_model

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
        
        # Define background processing task
        def process_and_index():
            try:
                # Process document (extract text and chunk)
                chunks = process_document(str(file_path), document_id)
                
                # Store original filename in metadata
                for chunk in chunks:
                    chunk["metadata"]["filename"] = file.filename
                
                # Index chunks in vector store
                add_documents(chunks)
                print(f"✅ Successfully indexed {len(chunks)} chunks from {file.filename}")
            except Exception as e:
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


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    documents = get_all_documents()
    return [DocumentInfo(**doc) for doc in documents]


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

