import re
from typing import Dict, List, Optional
from openai import OpenAI, AuthenticationError, APIError

from config import (
    ENABLE_RERANKER,
    FETCH_K,
    LLM_MODEL_NAME,
    MAX_SELF_HEAL_ITERATIONS,
    NO_ANSWER_MESSAGE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_TIMEOUT_SECONDS,
    RELEVANCE_THRESHOLD,
    TOP_K,
    TOP_K_RERANK,
)
from utils.chunking import extract_keywords
from utils.embeddings import embed_texts, load_embedding_model
from utils.prompt import build_prompt
from utils.reranker import load_reranker_model, rerank_passages
from utils.vector_store import (
    extract_entity_references,
    hybrid_search,
    load_embedding_model_name,
    load_index,
    load_metadata,
)


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Create the OpenRouter-compatible OpenAI client with dynamic key support."""
    key = api_key or OPENROUTER_API_KEY
    if not key or key == "YOUR_OPENROUTER_API_KEY_HERE":
        raise ValueError("Missing OpenRouter API Key. Please enter a valid key in the sidebar.")
    return OpenAI(
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        timeout=OPENROUTER_TIMEOUT_SECONDS,
    )


def _load_vector_store() -> Dict[str, object]:
    index = load_index()
    metadata = load_metadata()
    if index is None or index.ntotal == 0 or not metadata:
        raise FileNotFoundError(
            "The vector store is missing or empty. Please upload documents and click 'Process / Ingest All Documents' in the sidebar."
        )
    stored_model_name = load_embedding_model_name()
    if stored_model_name is None:
        stored_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return {
        "index": index,
        "metadata": metadata,
        "embedding_model_name": stored_model_name,
    }


def _encode_query(query: str, model_name: str):
    model = load_embedding_model(model_name)
    embeddings = embed_texts(model, [query])
    return embeddings[0]


def _rewrite_query_for_self_healing(query: str) -> str:
    """Rewrite or expand query by extracting keywords, rule/section numbers, and entity references for CRAG self-healing."""
    keywords = extract_keywords(query, max_keywords=8)
    entities = extract_entity_references(query)
    entity_tokens = [f"{e[0]} {e[1]}" for e in entities]

    rule_matches = re.findall(r"\b(rule|section|sub-section|chapter)\s+(\d+)\b", query, flags=re.IGNORECASE)
    rule_expansions = []
    for kind, num in rule_matches:
        rule_expansions.extend([f"{kind.capitalize()} {num}", f"{num}.", f"{kind.lower()}-{num}"])

    combined_tokens = set(keywords + entity_tokens + rule_expansions)
    if combined_tokens:
        return f"{query} {' '.join(combined_tokens)}"

    return query


def llm_refine_query(
    query: str,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Refine user query using LLM to fix typos/OCR noise, bridge vocabulary mismatches (synonyms),
    and add relevant search keywords while PRESERVING all exact numbers, IDs, and proper nouns.
    """
    rule_expanded = _rewrite_query_for_self_healing(query)
    if len(query.strip().split()) <= 2:
        return rule_expanded

    try:
        client = get_openai_client(api_key)
        target_model = llm_model or LLM_MODEL_NAME

        system_instruction = (
            "You are a search query optimizer for a document retrieval system. "
            "Your job is to refine the user's search query by:\n"
            "1. Fixing typos, spelling errors, or OCR noise (e.g. 'reciept' -> 'receipt', 'accunt' -> 'account').\n"
            "2. Adding relevant synonyms to solve vocabulary mismatches (e.g. 'revenue' -> 'turnover income').\n"
            "3. Preserving ALL exact numbers, account numbers, section/rule numbers, serial IDs, and proper nouns unchanged.\n"
            "Output ONLY the optimized, expanded search query as a single concise line of keywords. Do not include quotes, preamble, or explanations."
        )

        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Optimize search query: {query}"},
            ],
            temperature=0.0,
            max_tokens=60,
            timeout=5.0,
        )

        refined = str(response.choices[0].message.content or "").strip().strip('"').strip("'")
        if refined and len(refined) > 3:
            return f"{rule_expanded} {refined}"

    except Exception:
        pass

    return rule_expanded


def calculate_confidence_score(reranked_sources: List[Dict[str, object]]) -> int:
    """Calculate overall retrieval confidence score (0 - 100%)."""
    if not reranked_sources:
        return 0

    scores = []
    for src in reranked_sources:
        if "rerank_score" in src and src["rerank_score"] is not None:
            score = float(src["rerank_score"])
            norm = 1.0 / (1.0 + pow(2.71828, -score))  # sigmoid normalization
            scores.append(norm)
        elif "combined_score" in src:
            scores.append(float(src["combined_score"]))

    if not scores:
        return 50

    top_score = max(scores)
    avg_score = sum(scores) / len(scores)
    confidence = int((top_score * 0.7 + avg_score * 0.3) * 100)
    return min(99, max(10, confidence))


def _generate_answer(
    prompt: str,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    client = get_openai_client(api_key)
    target_model = llm_model or LLM_MODEL_NAME
    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert document assistant. Answer ONLY using the provided document excerpts. "
                        "Do not invent information, hallucinate, or rely on knowledge outside the supplied content."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=768,
        )
        text = response.choices[0].message.content
        return str(text or "").strip()
    except AuthenticationError:
        raise ValueError("OpenRouter Authentication Error (401). Please check your OpenRouter API key in the sidebar.")
    except Exception as exc:
        err_msg = str(exc)
        if "404" in err_msg or "not found" in err_msg.lower():
            raise ValueError(f"Model '{target_model}' was not found on OpenRouter. Please select 'google/gemini-2.5-flash' or 'openai/gpt-4o-mini' from the sidebar.")
        raise RuntimeError(f"OpenRouter API Error: {err_msg}")


def _select_answer_sources(
    query: str,
    candidates: List[Dict[str, object]],
    enable_reranker: bool = ENABLE_RERANKER,
) -> List[Dict[str, object]]:
    if not enable_reranker:
        return candidates[:TOP_K_RERANK]

    try:
        reranker = load_reranker_model()
        return rerank_passages(reranker, query, candidates, top_k=TOP_K_RERANK)
    except Exception:
        return candidates[:TOP_K_RERANK]


def answer_query(
    query: str,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_reranker: bool = ENABLE_RERANKER,
) -> Dict[str, object]:
    """
    Retrieve candidate passages, apply self-healing Corrective RAG (CRAG) loop if needed,
    rerank them, compute confidence score, and generate a grounded answer.
    """
    store = _load_vector_store()
    current_query = query
    iterations = 0
    was_self_healed = False
    candidates = []

    for iteration in range(1, MAX_SELF_HEAL_ITERATIONS + 1):
        iterations = iteration
        
        # Apply LLM Query Refinement & Expansion
        search_query = current_query
        if iteration == 1:
            search_query = llm_refine_query(current_query, llm_model=llm_model, api_key=api_key)

        query_embedding = _encode_query(search_query, store["embedding_model_name"])
        candidates = hybrid_search(
            query=search_query,
            query_embedding=query_embedding,
            index=store["index"],
            metadata=store["metadata"],
            top_k=FETCH_K,
        )

        if not candidates:
            if iteration < MAX_SELF_HEAL_ITERATIONS:
                current_query = _rewrite_query_for_self_healing(query)
                was_self_healed = True
                continue
            else:
                return {
                    "answer": NO_ANSWER_MESSAGE,
                    "sources": [],
                    "confidence_score": 0,
                    "self_healed": was_self_healed,
                    "iterations": iterations,
                }

        top_combined_score = max([c.get("combined_score", 0.0) for c in candidates[:5]] or [0.0])

        if top_combined_score < RELEVANCE_THRESHOLD and iteration < MAX_SELF_HEAL_ITERATIONS:
            current_query = _rewrite_query_for_self_healing(query)
            was_self_healed = True
        else:
            break

    reranked = _select_answer_sources(query, candidates, enable_reranker=enable_reranker)
    for rank, candidate in enumerate(reranked, start=1):
        candidate["rank"] = rank

    confidence_score = calculate_confidence_score(reranked)
    prompt = build_prompt(query, reranked)
    answer = _generate_answer(prompt, llm_model=llm_model, api_key=api_key)

    if NO_ANSWER_MESSAGE.lower() in answer.lower():
        confidence_score = min(confidence_score, 15)

    sources = [
        {
            "document_name": candidate.get("document_name") or candidate.get("source"),
            "source": candidate.get("source") or candidate.get("document_name"),
            "page_number": candidate.get("page_number", 1),
            "heading": candidate.get("heading"),
            "parent_headings": candidate.get("parent_headings"),
            "block_type": candidate.get("block_type"),
            "is_ocr": candidate.get("is_ocr", False),
            "rerank_score": candidate.get("rerank_score"),
            "combined_score": candidate.get("combined_score"),
            "text": candidate.get("text"),
        }
        for candidate in reranked
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence_score": confidence_score,
        "self_healed": was_self_healed,
        "iterations": iterations,
        "retrieval_query": search_query,
    }


class RAGSystem:
    """Wrapper class for Streamlit interface compatibility."""
    def answer_question(self, query: str, llm_model: Optional[str] = None, api_key: Optional[str] = None, enable_reranker: bool = ENABLE_RERANKER) -> Dict[str, object]:
        return answer_query(query, llm_model=llm_model, api_key=api_key, enable_reranker=enable_reranker)
