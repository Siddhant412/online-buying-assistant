from app.llm_answerer import generate_llm_answer
from app.providers import ChatResponse


def test_generate_llm_answer_uses_provider_and_enforces_citations(monkeypatch) -> None:
    class FakeProvider:
        def chat(self, system: str, user: str, model_name: str | None = None) -> ChatResponse:
            assert "careful product QA assistant" in system
            assert "[R1]" in user
            return ChatResponse(
                content="The coating lasted six months.",
                provider="fake",
                model=model_name or "fake-model",
                latency_ms=12,
            )

    monkeypatch.setattr("app.llm_answerer.get_llm_provider", lambda: FakeProvider())

    result = generate_llm_answer(
        "How long did the coating last?",
        [
            {
                "source_id": "R1",
                "source_type": "review",
                "text": "Great. Coating lasted 6 months with daily use.",
                "meta": {"rating": 4, "date": "2024-06-12", "verified": True},
            }
        ],
        model_name="fixture-model",
    )

    assert result["answer"] == "The coating lasted six months. [R1]"
    assert result["provider"] == "fake"
    assert result["model"] == "fixture-model"
    assert result["provider_latency_ms"] == 12
