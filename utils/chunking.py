import re
import uuid
from typing import Dict, List, Optional, Tuple

# Generic stop words for keyword extraction and noise filtering.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "when", "while",
    "for", "to", "in", "on", "at", "by", "with", "of", "from",
    "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "these", "those", "it", "its", "as", "such", "can",
    "will", "should", "could", "would", "may", "might", "has",
    "have", "had", "do", "does", "did", "not", "no", "yes",
}

DOCUMENT_TYPE_HINTS = [
    ("research paper", ["abstract", "introduction", "method", "results", "conclusion", "references"]),
    ("technical documentation", ["install", "configure", "usage", "api", "setup", "manual", "reference", "troubleshoot"]),
    ("manual", ["step", "steps", "instruction", "user manual", "installation"]),
    ("policy", ["policy", "compliance", "guideline", "procedure", "governance"]),
    ("story", ["chapter", "once upon a time", "character", "novel", "story"]),
    ("book", ["chapter", "preface", "epilogue", "publisher"]),
    ("report", ["report", "executive summary", "findings", "analysis", "overview"]),
    ("financial", ["balance sheet", "income statement", "revenue", "expense", "cash flow"]),
    ("medical", ["patient", "clinical", "diagnosis", "treatment", "medical", "health"]),
    ("faq", ["faq", "frequently asked", "question", "answer"]),
]

DEFAULT_DOCUMENT_TYPE = "document"
MIN_TOKENS = 12
TARGET_TOKENS = 520
MAX_TOKENS = 800
OVERLAP_RATIO = 0.12

HEADING_TYPES = {"heading", "title"}
STRUCTURED_BLOCK_TYPES = {"heading", "paragraph", "list", "table", "code", "quote", "caption", "image", "formula"}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sanitize_text(text: str) -> str:
    text = normalize_whitespace(text)
    return text


def estimate_tokens(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\s\w]", text)
    return len(tokens)


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if re.fullmatch(r"(?:page|pg|p\.?)(\s*\d+)(?:\s*/\s*\d+)?", lower):
        return True
    if re.fullmatch(r"(?:chapter|section|article)\s+\d+", lower):
        return False
    if any(token in lower for token in ["copyright", "all rights reserved", "confidential", "draft", "version"]):
        return True
    if lower in {"table of contents", "contents", "index", "back", "next", "previous", "home"}:
        return True
    return False


def remove_document_noise(text: str) -> str:
    lines = text.splitlines()
    cleaned: List[str] = []
    for line in lines:
        if is_noise_line(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def split_into_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[\.\?!])\s+(?=[A-Z0-9])", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def looks_like_heading(line: str) -> bool:
    text = line.strip()
    if len(text) < 4 or len(text) > 120:
        return False
    if text.endswith(".") or text.endswith("?") or text.endswith("!"):
        return False
    capitalized = sum(1 for token in text.split() if token[:1].isupper())
    return capitalized >= max(1, len(text.split()) // 2)


def detect_document_type(title: Optional[str], text: str) -> str:
    lower = " ".join([str(title or "").lower(), str(text or "").lower()])
    for label, hints in DOCUMENT_TYPE_HINTS:
        if any(hint in lower for hint in hints):
            return label
    return DEFAULT_DOCUMENT_TYPE


def parse_list_block(paragraph: str) -> Dict[str, object]:
    return {
        "block_type": "list",
        "text": normalize_whitespace(paragraph),
        "token_count": estimate_tokens(paragraph),
    }


def parse_code_block(paragraph: str) -> Dict[str, object]:
    return {
        "block_type": "code",
        "text": paragraph.rstrip(),
        "token_count": estimate_tokens(paragraph),
    }


def parse_quote_block(paragraph: str) -> Dict[str, object]:
    return {
        "block_type": "quote",
        "text": normalize_whitespace(paragraph.lstrip("> ")),
        "token_count": estimate_tokens(paragraph),
    }


def parse_caption_block(paragraph: str) -> Dict[str, object]:
    return {
        "block_type": "caption",
        "text": normalize_whitespace(paragraph),
        "token_count": estimate_tokens(paragraph),
    }


def parse_paragraph_block(paragraph: str) -> Dict[str, object]:
    return {
        "block_type": "paragraph",
        "text": normalize_whitespace(paragraph),
        "token_count": estimate_tokens(paragraph),
    }


def text_looks_like_table(paragraph: str) -> bool:
    lines = [line for line in paragraph.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if any("|" in line for line in lines):
        return True
    return False


def parse_blocks_from_text(text: str) -> List[Dict[str, object]]:
    if not text:
        return []
    text = remove_document_noise(text)
    segments = re.split(r"\n{2,}", text)
    blocks: List[Dict[str, object]] = []

    for raw in segments:
        paragraph = raw.strip()
        if not paragraph:
            continue
        if paragraph.startswith("```") or paragraph.startswith("~~~"):
            blocks.append(parse_code_block(paragraph))
            continue
        if paragraph.startswith(">"):
            blocks.append(parse_quote_block(paragraph))
            continue
        if paragraph.strip().startswith(('-', '*', '+')) and '\n' in paragraph:
            blocks.append(parse_list_block(paragraph))
            continue
        if text_looks_like_table(paragraph):
            blocks.append({
                "block_type": "table",
                "text": normalize_whitespace(paragraph),
                "token_count": estimate_tokens(paragraph),
            })
            continue
        first_line = paragraph.splitlines()[0].strip()
        if looks_like_heading(first_line) and len(first_line.split()) <= 10:
            blocks.append({
                "block_type": "heading",
                "heading_level": 1,
                "text": first_line,
                "token_count": estimate_tokens(first_line),
            })
            remainder = paragraph[len(first_line):].strip()
            if remainder:
                blocks.append(parse_paragraph_block(remainder))
            continue
        if re.match(r"^(figure|table|image|chart|diagram)\b", first_line.lower()):
            blocks.append(parse_caption_block(paragraph))
            continue
        blocks.append(parse_paragraph_block(paragraph))

    return blocks


def normalize_block(block: Dict[str, object]) -> Dict[str, object]:
    block = {**block}
    block_type = str(block.get("block_type", "paragraph")).lower()
    if block_type not in STRUCTURED_BLOCK_TYPES:
        block_type = "paragraph"
    block["block_type"] = block_type

    if block_type == "heading":
        block["heading_level"] = int(block.get("heading_level", 1) or 1)
        block["text"] = normalize_whitespace(block.get("text", ""))
        if not block["text"]:
            block["text"] = "Untitled"
    else:
        block["text"] = normalize_whitespace(block.get("text", ""))
    block["token_count"] = estimate_tokens(block["text"])
    return block


def build_document_tree(blocks: List[Dict[str, object]], document_title: Optional[str] = None) -> List[Dict[str, object]]:
    stack: List[Dict[str, object]] = []
    tree: List[Dict[str, object]] = []

    for block in blocks:
        block = normalize_block(block)
        if block["block_type"] == "heading":
            level = max(1, int(block.get("heading_level", 1)))
            while len(stack) >= level:
                stack.pop()
            stack.append(block)
            block["heading"] = block["text"]
            block["parent_headings"] = [item["text"] for item in stack[:-1]]
            block["heading_level"] = level
        else:
            block["heading_level"] = len(stack)
            block["parent_headings"] = [item["text"] for item in stack]
            block["heading"] = stack[-1]["text"] if stack else (document_title or "")
        tree.append(block)
    return tree


def group_paragraphs(blocks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    for block in blocks:
        if block["block_type"] == "paragraph" and merged and merged[-1]["block_type"] == "paragraph" and block["heading"] == merged[-1]["heading"]:
            merged[-1]["text"] = normalize_whitespace(merged[-1]["text"] + "\n\n" + block["text"])
            merged[-1]["token_count"] = estimate_tokens(merged[-1]["text"])
            continue
        merged.append(block)
    return merged


def split_large_block(block: Dict[str, object], max_tokens: int = MAX_TOKENS) -> List[Dict[str, object]]:
    text = block.get("text", "")
    if not text:
        return []
    if block["token_count"] <= max_tokens:
        return [block]

    if block["block_type"] == "code":
        lines = text.splitlines()
        pieces: List[Dict[str, object]] = []
        current: List[str] = []
        current_tokens = 0
        for line in lines:
            line_tokens = estimate_tokens(line)
            if current_tokens + line_tokens > max_tokens and current:
                chunk_text = "\n".join(current)
                pieces.append({**block, "text": chunk_text, "token_count": estimate_tokens(chunk_text)})
                current = []
                current_tokens = 0
            current.append(line)
            current_tokens += line_tokens
        if current:
            chunk_text = "\n".join(current)
            pieces.append({**block, "text": chunk_text, "token_count": estimate_tokens(chunk_text)})
        return pieces

    sentences = split_into_sentences(text)
    pieces = []
    current = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens and current:
            chunk_text = " ".join(current)
            pieces.append({**block, "text": chunk_text, "token_count": estimate_tokens(chunk_text)})
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunk_text = " ".join(current)
        pieces.append({**block, "text": chunk_text, "token_count": estimate_tokens(chunk_text)})
    return pieces


def calculate_overlap(text: str, target_tokens: int) -> str:
    tokens = re.findall(r"\w+|[^\s\w]", text)
    overlap_count = max(1, int(target_tokens * OVERLAP_RATIO))
    if overlap_count >= len(tokens):
        return text
    return " ".join(tokens[-overlap_count:])


def build_chunk_text(
    blocks: List[Dict[str, object]],
    document_name: str,
    document_title: Optional[str] = None,
) -> str:
    if not blocks:
        return ""
    context = [f"Document: {document_title or document_name}".strip()]
    heading = blocks[-1].get("heading") or document_title or document_name
    parent_headings = blocks[-1].get("parent_headings", [])
    if heading:
        context.append(f"Heading: {heading}")
    if parent_headings:
        context.append(f"Parent headings: {' > '.join(parent_headings)}")
    context.append("Content:")
    content_lines = []
    for block in blocks:
        if block["block_type"] == "list":
            content_lines.append(block["text"])
        elif block["block_type"] == "table":
            content_lines.append(block["text"])
        elif block["block_type"] == "code":
            content_lines.append(block["text"])
        else:
            content_lines.append(block["text"])
    context.append("\n\n".join(content_lines))
    return "\n".join(context).strip()


def summarize_text(text: str) -> str:
    sentences = split_into_sentences(text)
    if sentences:
        summary = sentences[0]
        if len(summary.split()) < 8 and len(sentences) > 1:
            summary = f"{summary} {sentences[1]}"
        return normalize_whitespace(summary)
    return normalize_whitespace(text)[:200]


def extract_keywords(text: str, max_keywords: int = 6) -> List[str]:
    words = re.findall(r"\w+", str(text or "").lower())
    candidates = [word for word in words if word not in STOPWORDS and (len(word) > 2 or word.isdigit())]
    if not candidates:
        return []
    frequencies: Dict[str, int] = {}
    for word in candidates:
        frequencies[word] = frequencies.get(word, 0) + 1
    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:max_keywords]]


def validate_chunk_text(text: str) -> bool:
    if not text or not text.strip():
        return False
    token_count = estimate_tokens(text)
    if token_count < MIN_TOKENS:
        return False
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", text)
    if cleaned and len(cleaned) / max(1, len(text)) < 0.15:
        return False
    if re.fullmatch(r"\s*[0-9\W]+\s*", text):
        return False
    return True


def select_dominant_block_type(group: List[Dict[str, object]]) -> str:
    if not group:
        return "paragraph"
    weights = {}
    for block in group:
        block_type = block.get("block_type", "paragraph")
        weights[block_type] = weights.get(block_type, 0) + block.get("token_count", 0)
    return max(weights.items(), key=lambda item: (item[1], item[0]))[0]


def build_chunk_metadata(
    chunk_text: str,
    document_name: str,
    source: str,
    page_number: int,
    is_ocr: bool,
    ocr_engine: Optional[str],
    block_type: str,
    document_type: str,
    heading: Optional[str],
    parent_headings: List[str],
    heading_level: int,
) -> Dict[str, object]:
    token_count = estimate_tokens(chunk_text)
    summary = summarize_text(chunk_text)
    keywords = extract_keywords(chunk_text)
    return {
        "chunk_id": str(uuid.uuid4()),
        "document_name": document_name,
        "source": source,
        "page_number": page_number,
        "is_ocr": is_ocr,
        "ocr_engine": ocr_engine,
        "document_type": document_type,
        "language": "en",
        "block_type": block_type,
        "heading": heading or "",
        "parent_headings": parent_headings,
        "heading_level": heading_level,
        "chunk_summary": summary,
        "keywords": keywords,
        "token_count": token_count,
        "text": chunk_text,
    }


def merge_blocks_for_chunks(blocks: List[Dict[str, object]], target_tokens: int) -> List[List[Dict[str, object]]]:
    chunks: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = []
    current_tokens = 0

    for block in blocks:
        pieces = split_large_block(block, MAX_TOKENS)
        for piece in pieces:
            if current and current_tokens + piece["token_count"] > target_tokens and current_tokens >= int(target_tokens * 0.5):
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(piece)
            current_tokens += piece["token_count"]
            if current_tokens >= target_tokens:
                chunks.append(current)
                current = []
                current_tokens = 0

    if current:
        chunks.append(current)
    return chunks


def apply_chunk_overlap(chunks: List[List[Dict[str, object]]]) -> List[List[Dict[str, object]]]:
    overlapped: List[List[Dict[str, object]]] = []
    previous_text = ""
    for chunk_blocks in chunks:
        chunk_blocks = list(chunk_blocks)
        overlap_text = calculate_overlap(previous_text, TARGET_TOKENS) if previous_text else ""
        if overlap_text:
            overlap_block = {
                "block_type": "paragraph",
                "text": overlap_text,
                "token_count": estimate_tokens(overlap_text),
                "heading": chunk_blocks[0].get("heading", ""),
                "parent_headings": chunk_blocks[0].get("parent_headings", []),
                "heading_level": chunk_blocks[0].get("heading_level", 0),
            }
            chunk_blocks = [overlap_block] + chunk_blocks
        chunk_text = "\n\n".join(item["text"] for item in chunk_blocks if item.get("text"))
        previous_text = chunk_text
        overlapped.append(chunk_blocks)
    return overlapped


def remove_duplicate_chunks(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    unique: List[Dict[str, object]] = []
    seen = set()
    for chunk in chunks:
        signature = normalize_whitespace(chunk.get("text", "")).lower()
        if not signature or signature in seen:
            continue
        seen.add(signature)
        unique.append(chunk)
    return unique


def chunk_document_blocks(
    document_name: str,
    blocks: List[Dict[str, object]],
    source_filename: Optional[str] = None,
    page_number: int = 1,
    is_ocr: bool = False,
    ocr_engine: Optional[str] = None,
    document_title: Optional[str] = None,
) -> List[Dict[str, object]]:
    source = source_filename or document_name
    document_title = document_title or document_name
    blocks = [normalize_block(block) for block in blocks]
    blocks = build_document_tree(blocks, document_title=document_title)
    blocks = group_paragraphs(blocks)
    document_type = detect_document_type(document_title, " ".join(block["text"] for block in blocks))
    target_tokens = adapt_target_tokens(document_type)
    block_groups = merge_blocks_for_chunks(blocks, target_tokens)
    block_groups = apply_chunk_overlap(block_groups)

    chunks: List[Dict[str, object]] = []
    for group in block_groups:
        chunk_text = build_chunk_text(group, document_name=document_name, document_title=document_title)
        if not validate_chunk_text(chunk_text):
            continue
        dominant_block_type = select_dominant_block_type(group)
        heading = group[-1].get("heading")
        parent_headings = group[-1].get("parent_headings", [])
        heading_level = group[-1].get("heading_level", 0)
        metadata = build_chunk_metadata(
            chunk_text=chunk_text,
            document_name=document_name,
            source=source,
            page_number=page_number,
            is_ocr=is_ocr,
            ocr_engine=ocr_engine,
            block_type=dominant_block_type,
            document_type=document_type,
            heading=heading,
            parent_headings=parent_headings,
            heading_level=heading_level,
        )
        chunks.append(metadata)
    return remove_duplicate_chunks(chunks)


def adapt_target_tokens(document_type: str) -> int:
    if document_type in {"story", "book"}:
        return 620
    if document_type in {"manual", "policy", "report", "financial", "medical"}:
        return 500
    if document_type == "faq":
        return 420
    return TARGET_TOKENS


def chunk_pages(page_records: List[Dict[str, object]], source_filename: str) -> List[Dict[str, object]]:
    all_chunks: List[Dict[str, object]] = []
    for page in page_records:
        page_number = int(page.get("page_number", 1))
        is_ocr = bool(page.get("is_ocr", False))
        ocr_engine = page.get("ocr_engine") if is_ocr else None
        title = page.get("title") or source_filename
        document_name = page.get("document_name") or source_filename
        structured = page.get("structured_blocks")
        if structured:
            blocks = [normalize_block(block) for block in structured]
        else:
            text = page.get("text", "")
            blocks = parse_blocks_from_text(text)
        if not blocks:
            continue
        chunks = chunk_document_blocks(
            document_name=document_name,
            blocks=blocks,
            source_filename=source_filename,
            page_number=page_number,
            is_ocr=is_ocr,
            ocr_engine=ocr_engine,
            document_title=title,
        )
        all_chunks.extend(chunks)
    return remove_duplicate_chunks(all_chunks)


# Compatibility alias for existing ingestion code.
chunk_pdf_pages = chunk_pages
