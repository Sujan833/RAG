import pytest
from pathlib import Path
from utils.document_parsers import is_noise_line, normalize_ocr_text, remove_noise_from_text

def test_is_noise_line():
    assert is_noise_line("Page 1 of 10") is True
    assert is_noise_line("Copyright 2026") is True
    assert is_noise_line("Important Bank Account Information") is False

def test_normalize_ocr_text():
    raw = "AccountNo:32910100010444\nIFSCode:BARB0VADLAM"
    normalized = normalize_ocr_text(raw)
    assert "Account No" in normalized or "Account" in normalized
    assert "IFS Code" in normalized or "IFS" in normalized

def test_remove_noise_from_text():
    text = "Page 1\nValid Bank Statement Content\nConfidential"
    cleaned = remove_noise_from_text(text)
    assert "Valid Bank Statement Content" in cleaned
    assert "Page 1" not in cleaned
