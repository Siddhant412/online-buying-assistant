from __future__ import annotations

import math
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import requests

from .settings import get_settings

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized embedding per input text."""


class RerankerProvider(Protocol):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage."""


@dataclass(frozen=True)
class ChatResponse:
    content: str
    provider: str
    model: str
    latency_ms: int
    raw: dict[str, Any] | None = None


class LLMProvider(Protocol):
    def chat(self, system: str, user: str, model_name: str | None = None) -> ChatResponse:
        """Return a single chat completion."""


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1}


def lexical_relevance(query: str, passage: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    passage_tokens = _tokens(passage)
    overlap = query_tokens & passage_tokens
    return len(overlap) / math.sqrt(max(1, len(query_tokens)) * max(1, len(passage_tokens)))


def _torch_device_available(device: str) -> bool:
    if device == "cpu":
        return True
    try:
        import torch
    except Exception:
        return False
    if device == "mps":
        return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    if device == "cuda":
        return bool(torch.cuda.is_available())
    return False


def _resolve_device(requested: str, default: str = "cpu") -> str:
    device = (requested or default).strip().lower()
    if device not in {"cpu", "mps", "cuda"}:
        device = default
    return device if _torch_device_available(device) else "cpu"


class LocalSentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.emb_model_name
        self.device = _resolve_device(device or settings.emb_device)
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings. "
                    "Install the `rag` extra or disable embedding writes."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]


class LocalCrossEncoderRerankerProvider:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_name
        self.device = _resolve_device(device or settings.rerank_device)
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:
                raise RuntimeError(
                    "sentence-transformers is required for local reranking. "
                    "Install the `rag` extra or disable reranking."
                ) from exc
            try:
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception:
                if self.device == "cpu":
                    raise
                self.device = "cpu"
                self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        model = self._load_model()
        scores = model.predict([(query, passage) for passage in passages], convert_to_numpy=True)
        return [float(score) for score in scores]


class LexicalRerankerProvider:
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [lexical_relevance(query, passage) for passage in passages]


class OllamaChatProvider:
    provider_name = "ollama"

    def chat(self, system: str, user: str, model_name: str | None = None) -> ChatResponse:
        settings = get_settings()
        model = model_name or settings.ollama_model
        endpoint = f"{settings.ollama_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
        }
        started = perf_counter()
        response = requests.post(endpoint, json=payload, timeout=60)
        latency_ms = int((perf_counter() - started) * 1000)
        response.raise_for_status()
        data = response.json()
        return ChatResponse(
            content=data.get("message", {}).get("content", "").strip(),
            provider=self.provider_name,
            model=model,
            latency_ms=latency_ms,
            raw=data,
        )


class OpenAICompatibleChatProvider:
    provider_name = "openai_compatible"

    def chat(self, system: str, user: str, model_name: str | None = None) -> ChatResponse:
        settings = get_settings()
        base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        model = model_name or settings.openai_model
        if not model:
            raise RuntimeError("OPENAI_MODEL must be configured for the OpenAI-compatible provider.")

        headers = {"Content-Type": "application/json"}
        if settings.openai_api_key:
            headers["Authorization"] = f"Bearer {settings.openai_api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        started = perf_counter()
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        latency_ms = int((perf_counter() - started) * 1000)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        return ChatResponse(
            content=str(content).strip(),
            provider=self.provider_name,
            model=model,
            latency_ms=latency_ms,
            raw=data,
        )


def get_embedding_provider() -> EmbeddingProvider:
    return LocalSentenceTransformerEmbeddingProvider()


def get_reranker_provider() -> RerankerProvider:
    settings = get_settings()
    if not settings.enable_reranking:
        return LexicalRerankerProvider()
    return LocalCrossEncoderRerankerProvider()


def get_llm_provider() -> LLMProvider:
    provider = get_settings().llm_provider.strip().lower().replace("-", "_")
    if provider == "ollama":
        return OllamaChatProvider()
    if provider in {"openai", "openai_compatible", "openai_compat"}:
        return OpenAICompatibleChatProvider()
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")
