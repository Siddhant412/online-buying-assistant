from app.answerer import answer_from_passages


def test_answer_from_passages_reports_month_range() -> None:
    passages = [
        {
            "source_id": "R1",
            "source_type": "review",
            "text": "Great. Coating lasted 6 months with daily use.",
            "meta": {"rating": 4, "date": "2024-06-12", "verified": True, "helpful_votes": 18},
        },
        {
            "source_id": "R2",
            "source_type": "review",
            "text": "Peeling. Non-stick wore off in 3 months.",
            "meta": {"rating": 2, "date": "2024-11-20", "verified": True, "helpful_votes": 45},
        },
    ]

    result = answer_from_passages("How long does the coating last?", passages)

    assert result["answer"].startswith("Reports vary: about 3")
    assert "6 months based on user reviews." in result["answer"]
    assert result["confidence"] == 0.55
    assert [c["source_id"] for c in result["citations"]] == ["R1", "R2"]


def test_answer_from_passages_handles_empty_evidence() -> None:
    result = answer_from_passages("Does it support induction?", [])

    assert result == {"answer": "Insufficient evidence.", "citations": [], "confidence": 0.0}
