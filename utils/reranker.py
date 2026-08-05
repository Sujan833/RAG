import os
from functools import lru_cache
from typing import Dict, List, Optional

import utils.patch_torchvision  # noqa: F401
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL_NAME


@lru_cache(maxsize=1)
def load_reranker_model() -> Optional[CrossEncoder]:
    """Load the CPU cross-encoder with in-memory singleton caching and online fallback."""
    # 1. Try loading from local HuggingFace cache first
    try:
        return CrossEncoder(
            RERANKER_MODEL_NAME,
            device="cpu",
            local_files_only=True,
        )
    except Exception:
        pass

    # 2. Try loading online from HuggingFace Hub
    try:
        return CrossEncoder(
            RERANKER_MODEL_NAME,
            device="cpu",
            local_files_only=False,
        )
    except Exception:
        return None


def rerank_passages(
    reranker: Optional[CrossEncoder],
    query: str,
    candidates: List[Dict[str, object]],
    top_k: int,
) -> List[Dict[str, object]]:
    """Score and return the highest-ranked passage candidates."""
    if reranker is None or not candidates:
        return candidates[:top_k]

    try:
        pairs = [[query, candidate["text"]] for candidate in candidates]
        scores = reranker.predict(pairs, show_progress_bar=False, batch_size=16)
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        sorted_candidates = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
        return sorted_candidates[:top_k]
    except Exception:
        return candidates[:top_k]
