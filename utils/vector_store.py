import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# Lock FAISS OpenMP threading to 1 thread on Windows to avoid DLL memory collisions
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

from config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    BM25_K1,
    BM25_B,
    HYBRID_BM25_TOP_K,
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_TOP_K,
    HYBRID_VECTOR_WEIGHT,
)


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", str(text or "").lower())


BM25_INDEX_PATH = Path(FAISS_INDEX_PATH).parent / "bm25_index.pkl"
EMBEDDING_MODEL_NAME_PATH = Path(FAISS_INDEX_PATH).parent / "embedding_model.txt"


def save_embedding_model_name(model_name: str) -> None:
    with open(EMBEDDING_MODEL_NAME_PATH, "w", encoding="utf-8") as f:
        f.write(model_name)


def load_embedding_model_name() -> Optional[str]:
    if not EMBEDDING_MODEL_NAME_PATH.exists():
        return None
    return EMBEDDING_MODEL_NAME_PATH.read_text(encoding="utf-8").strip()


def save_index(index: faiss.Index) -> None:
    index_path = Path(FAISS_INDEX_PATH)
    faiss.write_index(index, str(index_path))


def load_index() -> Optional[faiss.Index]:
    index_path = Path(FAISS_INDEX_PATH)
    if not index_path.exists():
        return None
    return faiss.read_index(str(index_path))


def save_metadata(metadata: List[Dict[str, object]]) -> None:
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)


def load_metadata() -> List[Dict[str, object]]:
    if not Path(METADATA_PATH).exists():
        return []
    with open(METADATA_PATH, "rb") as f:
        return pickle.load(f)


def save_bm25_index(bm25: BM25Okapi) -> None:
    """Persist BM25 index to disk."""
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25_index() -> Optional[BM25Okapi]:
    """Load BM25 index from disk."""
    if not BM25_INDEX_PATH.exists():
        return None
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def build_bm25_index(metadata: List[Dict[str, object]]) -> BM25Okapi:
    """Build BM25 index from document texts and available document metadata."""
    corpus = [
        tokenize(
            " ".join(
                str(doc.get(field, ""))
                for field in [
                    "document_name", "source", "title", "document_type", "heading", "parent_headings", "block_type", "text",
                ]
            )
        )
        for doc in metadata
    ]
    bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
    save_bm25_index(bm25)
    return bm25


def build_faiss_index(embeddings: np.ndarray, metadata: List[Dict[str, object]], embedding_model_name: str) -> faiss.Index:
    """Build a new FAISS index and persist metadata."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    save_index(index)
    save_metadata(metadata)
    save_embedding_model_name(embedding_model_name)
    build_bm25_index(metadata)
    return index


def create_empty_index(dim: int) -> faiss.Index:
    """Create the FAISS index type used by this project."""
    return faiss.IndexFlatIP(dim)


def persist_vector_store(index: faiss.Index, metadata: List[Dict[str, object]], embedding_model_name: str) -> None:
    """Persist FAISS, metadata, embedding model name, and BM25 in one place."""
    save_index(index)
    save_metadata(metadata)
    save_embedding_model_name(embedding_model_name)
    build_bm25_index(metadata)


def append_to_index(
    index: faiss.Index,
    embeddings: np.ndarray,
    new_metadata: List[Dict[str, object]],
    existing_metadata: List[Dict[str, object]],
) -> faiss.Index:
    """Append new embeddings and metadata to an existing FAISS index."""
    index.add(embeddings)
    existing_metadata.extend(new_metadata)
    save_index(index)
    save_metadata(existing_metadata)
    build_bm25_index(existing_metadata)
    return index


def search_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS and return similarity scores and metadata indices."""
    if index is None or index.ntotal == 0:
        return np.array([]), np.array([])
    query_embedding = query_embedding.reshape(1, -1).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, ids = index.search(query_embedding, top_k)
    return scores[0], ids[0]


def search_bm25(query: str, top_k: int) -> List[Tuple[int, float]]:
    """Search using BM25 and return (metadata_id, score) tuples."""
    bm25 = load_bm25_index()
    if bm25 is None:
        return []
    
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    
    # Get top_k by score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in ranked[:top_k] if score > 0]


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {idx: 0.0 for idx in scores}
    return {idx: score / max_score for idx, score in scores.items()}


def normalize_for_matching(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def extract_definition_subject(query: str) -> str:
    """Extract the likely term from common definition-style questions."""
    normalized = normalize_for_matching(query).strip(" ?.")
    patterns = [
        r"^what\s+is\s+(.+)$",
        r"^what\s+are\s+(.+)$",
        r"^define\s+(.+)$",
        r"^meaning\s+of\s+(.+)$",
        r"^explain\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if match:
            subject = match.group(1)
            subject = re.sub(r"^(a|an|the)\s+", "", subject)
            if not subject.startswith("defined under "):
                return subject.strip(" ?.\"'")
    return ""


def extract_entity_references(query: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Extract generic document entity references across all document types.
    Matches: Rule 21, Section 5, Article 14, Clause 3, Chapter 2, Step 4, Item 7, Table 1, Figure 2, Schedule 5, Regulation 8, etc.
    Returns list of (entity_type, entity_number, clause_subnumber).
    """
    references: List[Tuple[str, str, Optional[str]]] = []
    pattern = (
        r"\b(rule|section|sec|article|art|clause|cl|regulation|reg|chapter|ch|schedule|item|table|figure|fig|step|part|order|policy|level)\s*"
        r"(\d+[a-z]?)\s*(?:\(\s*([0-9]+[a-z]?)\s*\))?"
    )
    for match in re.finditer(pattern, str(query or ""), flags=re.IGNORECASE):
        entity_type = match.group(1).lower()
        entity_num = match.group(2).lower()
        clause_num = match.group(3).lower() if match.group(3) else None
        references.append((entity_type, entity_num, clause_num))
    return references


def extract_meaningful_query_terms(query: str) -> List[str]:
    """Extract content-bearing query terms for exact lexical matching."""
    stop_terms = {
        "a", "an", "the", "and", "or", "but", "by", "for", "from", "in", "is", "of", "on", "to",
        "what", "whats", "which", "who", "where", "when", "why", "how", "given", "give", "tell",
    }
    terms = []
    for term in tokenize(query):
        if term in stop_terms:
            continue
        if len(term) < 3 and not term.isdigit():
            continue
        terms.append(term)
    return terms


def query_phrase_boost(query: str, blob: str) -> float:
    """Boost exact user-intent phrases that dense search may dilute in large mixed corpora."""
    terms = extract_meaningful_query_terms(query)
    boost = 0.0

    if not terms:
        return boost

    if len(terms) >= 2 and all(re.search(rf"\b{re.escape(term)}\b", blob) for term in terms[:4]):
        boost += 0.7

    phrase_pairs = zip(terms, terms[1:])
    for first, second in phrase_pairs:
        if re.search(rf"\b{re.escape(first)}\s+{re.escape(second)}\b", blob):
            boost += 1.2

    return boost


def lexical_relevance_boost(query: str, item: Dict[str, object]) -> float:
    """Boost exact lexical signals that dense embeddings often under-rank across ALL document types."""
    heading = normalize_for_matching(item.get("heading", ""))
    parent_headings = normalize_for_matching(" ".join(str(value) for value in item.get("parent_headings", []) or []))
    text = normalize_for_matching(item.get("text", ""))
    blob = f"{heading} {parent_headings} {text}"

    boost = query_phrase_boost(query, blob)
    subject = extract_definition_subject(query)
    if subject:
        escaped_subject = re.escape(subject)
        numbered_definition = rf"\(\s*[0-9]+[a-z]?\s*\)\s*[\[0-9\s]*[\"“”]\s*{escaped_subject}\s*[\"“”][^.;]{{0,120}}\b(?:means|includes)\b"
        quoted_definition = rf"[\"“”]\s*{escaped_subject}\s*[\"“”][^.;]{{0,120}}\b(?:means|includes)\b"
        line_definition = rf"(?:^|[.;:])\s*{escaped_subject}\s+(?:means|includes)\b"
        if re.search(numbered_definition, blob):
            boost += 2.0
        elif re.search(quoted_definition, blob):
            boost += 1.4
        elif re.search(line_definition, blob):
            boost += 0.6
        elif re.search(rf"\b{escaped_subject}\b", blob):
            boost += 0.15
        if subject in heading:
            boost += 0.2

    for entity_type, entity_num, clause_num in extract_entity_references(query):
        exact_pattern = rf"\b{re.escape(entity_type)}\b[\s\-:\.#]*\b{re.escape(entity_num)}\b"
        if re.search(exact_pattern, blob):
            boost += 1.8
        elif re.search(rf"\b{re.escape(entity_num)}\b", blob) and entity_type in blob:
            boost += 0.8

        if clause_num:
            if re.search(rf"\(\s*{re.escape(clause_num)}\s*\)", blob):
                boost += 1.2
            if re.search(rf"\b{re.escape(clause_num)}\b", heading):
                boost += 0.3

    return boost


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    index: faiss.Index,
    metadata: List[Dict[str, object]],
    top_k: int,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    dense_weight: float = HYBRID_VECTOR_WEIGHT,
    vector_k: int = HYBRID_VECTOR_TOP_K,
    bm25_k: int = HYBRID_BM25_TOP_K,
) -> List[Dict[str, object]]:
    """
    Perform hybrid search combining BM25, dense retrieval, and universal lexical entity boosting.
    """
    candidate_k = min(len(metadata), max(top_k * 3, top_k))
    effective_bm25_k = max(bm25_k, candidate_k)
    effective_vector_k = max(vector_k, candidate_k)

    # Get BM25 results
    bm25_results = search_bm25(query, effective_bm25_k)
    bm25_dict = {idx: score for idx, score in bm25_results}
    
    # Get dense results
    dense_scores, dense_ids = search_index(index, query_embedding, effective_vector_k)
    dense_dict = {
        int(dense_id): float(dense_score)
        for dense_id, dense_score in zip(dense_ids, dense_scores)
        if dense_id >= 0
    }
    
    normalized_bm25 = normalize_scores(bm25_dict)
    normalized_dense = normalize_scores(dense_dict)

    # Combine results using the configured hybrid formula.
    combined_scores: Dict[int, float] = {}
    for idx in set(normalized_dense) | set(normalized_bm25):
        combined_scores[idx] = (
            dense_weight * normalized_dense.get(idx, 0.0)
            + bm25_weight * normalized_bm25.get(idx, 0.0)
        )
    
    # Sort by combined score and return top_k
    boosted_scores = {
        idx: score + lexical_relevance_boost(query, metadata[idx])
        for idx, score in combined_scores.items()
        if 0 <= idx < len(metadata)
    }
    sorted_results = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    
    for idx, boosted_score in sorted_results[:top_k]:
        if 0 <= idx < len(metadata):
            item = metadata[idx].copy()
            base_score = combined_scores.get(idx, 0.0)
            item["combined_score"] = boosted_score
            item["base_score"] = base_score
            item["lexical_boost"] = boosted_score - base_score
            item["bm25_score"] = bm25_dict.get(idx, 0.0)
            item["dense_score"] = dense_dict.get(idx, 0.0)
            results.append(item)
    
    return results
