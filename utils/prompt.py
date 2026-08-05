from typing import List, Dict

from config import MAX_PROMPT_TOKENS, NO_ANSWER_MESSAGE


def build_prompt(query: str, sources: List[Dict[str, object]]) -> str:
    """
    Build a retrieval-first prompt that forces the model to answer only from provided excerpts.
    Enhanced with better metadata and source tracking.
    """
    source_blocks = []
    for idx, source in enumerate(sources, start=1):
        metadata_parts = []
        if source.get("heading"):
            metadata_parts.append(f"Heading: {source.get('heading')}")
        parent_headings = source.get("parent_headings")
        if parent_headings:
            if isinstance(parent_headings, list):
                metadata_parts.append(f"Parent headings: {' > '.join(parent_headings)}")
            else:
                metadata_parts.append(f"Parent headings: {parent_headings}")
        if source.get("document_type"):
            metadata_parts.append(f"Document type: {source.get('document_type')}")
        block_type = source.get("block_type")
        if block_type and block_type != "paragraph":
            metadata_parts.append(f"Block type: {block_type}")
        metadata_info = f" | {' | '.join(metadata_parts)}" if metadata_parts else ""

        block = (
            f"Source {idx}: {source.get('document_name') or source.get('source')} | Page {source.get('page') or source.get('page_number')}{metadata_info}\n"
            f"Reranker Score: {source.get('rerank_score', 0):.3f}\n"
            f"---\n"
            f"{source['text']}"
        )
        source_blocks.append(block)
    
    prefix = (
        "You are a document assistant. Answer ONLY from the retrieved document excerpts. "
        "Do not use prior knowledge, infer facts, or guess answers.\n\n"
        "GROUNDING GUIDELINES:\n"
        "1. Answer ONLY using information explicitly present in the excerpts.\n"
        "2. Do not introduce or invent facts that are not in the provided text.\n"
        f"3. If the answer is not explicitly present in the excerpts, respond exactly: \"{NO_ANSWER_MESSAGE}\"\n"
        "4. Prefer quoted or clearly referenced passages from the excerpts over paraphrasing.\n"
        "5. Cite source metadata, including document and page, and include heading information when available.\n\n"
        "Document Context:\n"
    )
    suffix = (
        f"Question: {query}\n\n"
        "Answer (based strictly on the provided excerpts):"
    )

    available_context_chars = max(MAX_PROMPT_TOKENS - len(prefix) - len(suffix) - 4, 500)
    context_parts = []
    used_chars = 0
    for block in source_blocks:
        separator = "\n\n" if context_parts else ""
        remaining = available_context_chars - used_chars - len(separator)
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rsplit(" ", 1)[0].strip()
        if block:
            context_parts.append(block)
            used_chars += len(separator) + len(block)

    context = "\n\n".join(context_parts)
    return f"{prefix}{context}\n\n{suffix}"
