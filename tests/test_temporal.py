from datetime import date

from app.temporal import analyze_temporal_conflict


def test_analyze_temporal_conflict_groups_review_sentiment() -> None:
    passages = [
        {
            "source_id": "R1",
            "source_type": "review",
            "text": "Works well.",
            "meta": {"rating": 5, "date": "2025-03-01"},
        },
        {
            "source_id": "R2",
            "source_type": "review",
            "text": "Failed quickly.",
            "meta": {"rating": 1, "date": "2024-08-01"},
        },
        {
            "source_id": "S1",
            "source_type": "spec",
            "text": "Product spec.",
            "meta": {},
        },
    ]

    result = analyze_temporal_conflict(passages, today=date(2025, 4, 1))

    assert result["overall"] == {
        "n": 2,
        "pos": 1,
        "neg": 1,
        "neu": 0,
        "pos_rate": 0.5,
        "neg_rate": 0.5,
    }
    assert result["by_window"][0]["window"] == "0-6 mo"
    assert result["by_window"][0]["pos"] == 1
    assert result["by_window"][1]["window"] == "6-12 mo"
    assert result["by_window"][1]["neg"] == 1
