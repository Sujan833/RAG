import pytest
from utils.chunking import chunk_pages, chunk_pdf_pages, sanitize_text, estimate_tokens

def test_chunking_utilities():
    text = "  Section 1. Introduction to   GST RAG.  "
    sanitized = sanitize_text(text)
    assert sanitized == "Section 1. Introduction to GST RAG."
    assert estimate_tokens(sanitized) > 0

def test_chunk_pdf_pages():
    pages = [
        {"page_number": 1, "text": "Heading: Introduction\n" + ("Text content " * 40), "document_name": "test.pdf", "is_ocr": False},
        {"page_number": 2, "text": "Heading: Conclusion\n" + ("Final section " * 40), "document_name": "test.pdf", "is_ocr": True, "ocr_engine": "rapidocr"}
    ]
    chunks = chunk_pdf_pages(pages, source_filename="test.pdf")
    assert len(chunks) >= 1
    assert chunks[0]["document_name"] == "test.pdf"
