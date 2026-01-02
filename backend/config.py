"""
Configuration module for RAG Document Assistant.
Loads environment variables and provides configuration settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"

# Load environment variables from .env file in backend directory
env_path = BACKEND_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# LLM Provider: "ollama" or "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "phi3:mini")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Gemini API Keys (for fallback or if user prefers Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
_api_keys_str = os.getenv("GOOGLE_API_KEYS", "")

if _api_keys_str:
    GOOGLE_API_KEYS = [key.strip() for key in _api_keys_str.split(",") if key.strip()]
elif GOOGLE_API_KEY:
    GOOGLE_API_KEYS = [GOOGLE_API_KEY]
else:
    GOOGLE_API_KEYS = []

# Model Configuration (for Gemini)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Vector Store Configuration
COLLECTION_NAME = "documents"
TOP_K_RESULTS = 8

# Print configuration
print(f"🔧 LLM Provider: {LLM_PROVIDER.upper()}")
if LLM_PROVIDER == "ollama":
    print(f"🦙 Ollama Host: {OLLAMA_HOST}")
    print(f"🦙 Ollama LLM Model: {OLLAMA_LLM_MODEL}")
    print(f"🦙 Ollama Embed Model: {OLLAMA_EMBED_MODEL}")
else:
    print(f"🔑 Loaded {len(GOOGLE_API_KEYS)} Gemini API key(s)")

# Validate configuration
def validate_config():
    """Validate that required configuration is present."""
    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEYS:
        raise ValueError(
            "Gemini selected but no API keys configured. Set GOOGLE_API_KEYS or switch to LLM_PROVIDER=ollama"
        )
    return True
