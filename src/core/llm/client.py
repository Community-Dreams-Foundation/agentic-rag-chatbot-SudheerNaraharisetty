"""
LLM Client: Unified interface for OpenRouter (primary), NVIDIA NIM (embeddings fallback), and Groq (LLM fallback).

Provider Priority:
  - LLM: OpenRouter (Kimi K2.5) → Groq (Llama 3.1)
  - Embeddings: OpenRouter (Qwen3 8B) → NVIDIA NIM (llama-3.2-nv-embedqa)

Key Features:
  - No RPM rate limiting for OpenRouter (credit-based)
  - RPM throttling for NVIDIA NIM fallback only
  - Automatic provider failover
  - Streaming support
"""

import logging
import time
from typing import Any, Dict, Generator, List, Optional

import openai
from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client with:
    - OpenRouter (primary): Kimi K2.5 + Qwen3 Embedding 8B
    - NVIDIA NIM (fallback): Embeddings only
    - Groq (fallback): LLM only

    OpenRouter is credit-based (no RPM limits), NVIDIA NIM has 40 RPM limit.
    """

    def __init__(self):
        self.settings = get_settings()

        # Primary: OpenRouter client
        self.openrouter_client: Optional[openai.OpenAI] = None
        if self.settings.openrouter_api_key:
            self.openrouter_client = openai.OpenAI(
                base_url=self.settings.openrouter_base_url,
                api_key=self.settings.openrouter_api_key,
            )
            self.openrouter_model = self.settings.openrouter_model
            self.openrouter_embedding_model = self.settings.openrouter_embedding_model
            logger.info("OpenRouter client initialized")

        # Fallback: NVIDIA NIM client for embeddings
        self.nvidia_client: Optional[openai.OpenAI] = None
        if self.settings.nvidia_api_key:
            self.nvidia_client = openai.OpenAI(
                base_url=self.settings.nvidia_base_url,
                api_key=self.settings.nvidia_api_key,
            )
            self.nvidia_embedding_model = self.settings.nvidia_embedding_model
            logger.info("NVIDIA NIM client initialized (embeddings fallback)")

        # Fallback: Groq client for LLM
        self.groq_client: Optional[Groq] = None
        if self.settings.groq_api_key:
            self.groq_client = Groq(api_key=self.settings.groq_api_key)
            self.groq_model = self.settings.groq_model
            logger.info("Groq client initialized (LLM fallback)")

        # Rate-limit tracking (only for NVIDIA NIM fallback)
        self._request_timestamps: List[float] = []
        self._rpm_limit = self.settings.api_requests_per_minute
        self._batch_delay = self.settings.api_batch_delay_seconds

    def _throttle_nvidia(self):
        """Enforce RPM rate limit for NVIDIA NIM only."""
        now = time.time()
        window_start = now - 60.0
        self._request_timestamps = [
            t for t in self._request_timestamps if t > window_start
        ]
        if len(self._request_timestamps) >= self._rpm_limit:
            sleep_time = 60.0 - (now - self._request_timestamps[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"NVIDIA rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._request_timestamps.append(time.time())

    # ── Chat Completion ──────────────────────────────────────────────

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "openrouter",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Generate chat completion with automatic OpenRouter → Groq fallback.

        Args:
            messages: Conversation messages
            model: "openrouter" or "groq"
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            stream: Stream response chunks
        """
        if model == "openrouter":
            try:
                return self._openrouter_completion(
                    messages, temperature, max_tokens, stream, **kwargs
                )
            except Exception as e:
                logger.warning(f"OpenRouter failed ({e}), falling back to Groq")
                if self.groq_client:
                    return self._groq_completion(
                        messages, min(temperature, 0.9), max_tokens, stream, **kwargs
                    )
                raise
        elif model == "groq":
            return self._groq_completion(
                messages, temperature, max_tokens, stream, **kwargs
            )
        else:
            raise ValueError(f"Unknown model provider: {model}")

    def _openrouter_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        **kwargs,
    ):
        """Generate completion using OpenRouter (Kimi K2.5)."""
        if not self.openrouter_client:
            raise ValueError("OpenRouter client not available. Set OPENROUTER_API_KEY.")

        # No RPM throttling for OpenRouter - it's credit-based
        completion = self.openrouter_client.chat.completions.create(
            model=self.openrouter_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        if stream:
            return self._stream_openai_response(completion)
        else:
            return completion.choices[0].message.content

    def _groq_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        **kwargs,
    ):
        """Generate completion using Groq (fallback)."""
        if not self.groq_client:
            raise ValueError("Groq client not available. Set GROQ_API_KEY.")

        completion = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            return self._stream_groq_response(completion)
        else:
            return completion.choices[0].message.content

    def _stream_openai_response(self, completion) -> Generator[str, None, None]:
        """Stream OpenAI-compatible response (OpenRouter)."""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def _stream_groq_response(self, completion) -> Generator[str, None, None]:
        """Stream Groq response."""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── Embeddings with Fallback ─────────────────────────────────────

    def get_embeddings(
        self,
        texts: List[str],
        input_type: str = "query",
    ) -> List[List[float]]:
        """
        Get embeddings with OpenRouter (primary) → NVIDIA NIM (fallback).

        Args:
            texts: Texts to embed
            input_type: "query" for search queries, "passage" for documents

        Returns:
            List of embedding vectors (4096-dim for OpenRouter, 2048-dim for NVIDIA)
        """
        if not texts:
            return []

        batch_size = self.settings.embedding_batch_size
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Try OpenRouter first
            try:
                batch_embeddings = self._embed_openrouter(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(f"OpenRouter embeddings failed ({e}), trying NVIDIA NIM")
                batch_embeddings = self._embed_nvidia(batch)
                all_embeddings.extend(batch_embeddings)

            # Throttle between batches for NVIDIA fallback
            if i + batch_size < len(texts):
                time.sleep(self._batch_delay)

        return all_embeddings

    def _embed_openrouter(self, texts: List[str]) -> List[List[float]]:
        """Embed using OpenRouter (Qwen3 8B, 4096-dim)."""
        if not self.openrouter_client:
            raise ValueError("OpenRouter client not available")

        # No RPM throttling for OpenRouter
        response = self.openrouter_client.embeddings.create(
            model=self.openrouter_embedding_model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        reraise=True,
    )
    def _embed_nvidia(self, texts: List[str]) -> List[List[float]]:
        """Embed using NVIDIA NIM (fallback, 2048-dim)."""
        if not self.nvidia_client:
            raise ValueError("NVIDIA NIM client not available")

        self._throttle_nvidia()

        response = self.nvidia_client.embeddings.create(
            model=self.nvidia_embedding_model,
            input=texts,
            encoding_format="float",
            extra_body={
                "input_type": "passage",
                "truncate": "NONE",
            },
        )
        return [item.embedding for item in response.data]

    # ── Health Check ─────────────────────────────────────────────────

    def health_check(self) -> Dict[str, bool]:
        """Check connectivity to all LLM providers."""
        health = {
            "openrouter": False,
            "nvidia": False,
            "groq": False,
            "embeddings": False,
        }

        if self.openrouter_client:
            try:
                self.openrouter_client.models.list()
                health["openrouter"] = True
                health["embeddings"] = True  # OpenRouter provides embeddings
            except Exception:
                pass

        if self.nvidia_client:
            try:
                self.nvidia_client.models.list()
                health["nvidia"] = True
                if not health["embeddings"]:
                    health["embeddings"] = True  # NVIDIA can also provide embeddings
            except Exception:
                pass

        if self.groq_client:
            try:
                self.groq_client.models.list()
                health["groq"] = True
            except Exception:
                pass

        return health

    # ── Legacy Compatibility ────────────────────────────────────────

    def get_active_embedding_dimension(self) -> int:
        """Return the embedding dimension for the active provider."""
        if self.openrouter_client:
            return self.settings.openrouter_embedding_dimension
        return self.settings.nvidia_embedding_dimension
