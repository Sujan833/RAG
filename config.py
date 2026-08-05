import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR
VECTORDB_DIR = BASE_DIR / "vectordb"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
NO_ANSWER_MESSAGE = "I could not find this information in the provided documents."

# Default embedding model & reranker settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").strip().lower() in {"1", "true", "yes", "on"}

# OpenRouter Available Valid LLMs
AVAILABLE_LLMS = [
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-r1-distill-llama-70b",
]
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemini-2.5-flash")
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "15"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"]

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "140"))
TOP_K = int(os.getenv("TOP_K", "20"))
FETCH_K = int(os.getenv("FETCH_K", "60"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "6"))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "6500"))

# Self-Healing Corrective RAG (CRAG) Parameters
MAX_SELF_HEAL_ITERATIONS = 2
RELEVANCE_THRESHOLD = 0.35

# BM25 Configuration for hybrid retrieval
BM25_K1 = 1.5
BM25_B = 0.75
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.65"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.35"))
HYBRID_VECTOR_TOP_K = int(os.getenv("HYBRID_VECTOR_TOP_K", "35"))
HYBRID_BM25_TOP_K = int(os.getenv("HYBRID_BM25_TOP_K", "35"))
BM25_TOP_K = HYBRID_BM25_TOP_K

# Confidence thresholding
MIN_RETRIEVAL_SCORE = 0.25
MIN_RERANK_SCORE = 0.0

FAISS_INDEX_PATH = VECTORDB_DIR / "index.faiss"
METADATA_PATH = VECTORDB_DIR / "metadata.pkl"
