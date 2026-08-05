import pytest
from rag import _rewrite_query_for_self_healing, calculate_confidence_score

def test_rewrite_query_for_self_healing():
    query = "explain rule 20 under GST registration"
    rewritten = _rewrite_query_for_self_healing(query)
    assert "Rule 20" in rewritten or "20." in rewritten
    assert "explain" in rewritten

def test_calculate_confidence_score():
    sources = [
        {"rerank_score": 2.5, "combined_score": 0.85},
        {"rerank_score": 1.8, "combined_score": 0.70}
    ]
    score = calculate_confidence_score(sources)
    assert 50 <= score <= 99
