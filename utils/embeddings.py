import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator
from typing import List

import utils.patch_torchvision  # noqa: F401

# Limit CPU threads to reduce memory pressure when embedding large documents.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME


def _resolve_local_snapshot(model_name: str) -> str:
    """Return a local Hugging Face snapshot path for a cached model id."""
    if Path(model_name).exists():
        return model_name

    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_cache = cache_root / f"models--{model_name.replace('/', '--')}"
    refs_main = model_cache / "refs" / "main"
    snapshots_dir = model_cache / "snapshots"

    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = snapshots_dir / revision
        if snapshot.exists():
            return str(snapshot)

    if snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            return str(max(snapshots, key=lambda path: path.stat().st_mtime))

    raise RuntimeError(
        f"Embedding model '{model_name}' is not available locally. "
        "Use the sidebar button to download it, then process the PDFs again with that same model."
    )


@contextmanager
def _huggingface_offline(enabled: bool) -> Iterator[None]:
    """Temporarily prevent Hugging Face libraries from making network calls."""
    if not enabled:
        yield
        return

    previous_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous_hf_hub_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_hf_hub_offline
        if previous_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers_offline


@lru_cache(maxsize=8)
def _load_embedding_model_cached(model_name: str, local_files_only: bool) -> SentenceTransformer:
    """Load a sentence-transformer model on CPU."""
    model_name_or_path = _resolve_local_snapshot(model_name) if local_files_only else model_name
    try:
        with _huggingface_offline(local_files_only):
            return SentenceTransformer(
                model_name_or_path,
                device="cpu",
                local_files_only=local_files_only,
            )
    except TypeError:
        with _huggingface_offline(local_files_only):
            return SentenceTransformer(model_name_or_path, device="cpu")
    except OSError as exc:
        if local_files_only:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not available locally. "
                "Use the sidebar button to download it, or select the same model "
                "that was used when you processed the PDFs."
            ) from exc
        raise


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    *,
    local_files_only: bool = True,
) -> SentenceTransformer:
    """Load an embedding model without hidden network retries by default."""
    return _load_embedding_model_cached(model_name, local_files_only)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Embed text passages and normalize embeddings for FAISS cosine similarity."""
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    return embeddings.astype("float32")
