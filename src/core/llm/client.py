"""
LLM Client: Unified interface for NVIDIA NIM (Kimi K2.5) and Groq (fallback).
Implements batched embeddings, rate-limit throttling, and automatic failover.
"""

import time
import logging
from typing import Generator, Dict, Any, Optional, List

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
    - Kimi K2.5 thinking mode via NVIDIA NIM (primary)
    - Groq Llama 3.1 auto-fallback on NVIDIA failure
    - Batched embedding with throttle to respect NIM 40 RPM limit
    - Separate embedding API key support
    """

    def __init__(self):
        self.settings = get_settings()

        # Primary: NVIDIA NIM client for Kimi K2.5
        self.nvidia_client = openai.OpenAI(
            base_url=self.settings.nvidia_base_url,
            api_key=self.settings.nvidia_api_key,
        )
        self.nvidia_model = self.settings.nvidia_model

        # Embedding client (separate key for dedicated quota)
        self.embedding_client = openai.OpenAI(
            base_url=self.settings.nvidia_base_url,
            api_key=(
                self.settings.nvidia_embedding_api_key
                or self.settings.nvidia_api_key
            ),
        )
        self.embedding_model = self.settings.embedding_model
        self.embedding_dimension = self.settings.embedding_dimension

        # Fallback: Groq client
        self.groq_client = None
        if self.settings.groq_api_key:
            self.groq_client = Groq(api_key=self.settings.groq_api_key)
            self.groq_model = self.settings.groq_model

        # Rate-limit tracking
        self._request_timestamps: List[float] = []
        self._rpm_limit = self.settings.api_requests_per_minute
        self._batch_delay = self.settings.api_batch_delay_seconds

    def _throttle(self):
        """Enforce RPM rate limit by sleeping when necessary."""
        now = time.time()
        window_start = now - 60.0
        self._request_timestamps = [
            t for t in self._request_timestamps if t > window_start
        ]
        if len(self._request_timestamps) >= self._rpm_limit:
            sleep_time = 60.0 - (now - self._request_timestamps[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._request_timestamps.append(time.time())

    # ── Chat Completion ──────────────────────────────────────────────

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "nvidia",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        thinking: Optional[bool] = None,
        **kwargs,
    ) -> Any:
        """
        Generate chat completion with automatic NVIDIA → Groq fallback.

        Args:
            messages: Conversation messages
            model: "nvidia" or "groq"
            temperature: Sampling temperature (1.0 recommended for K2.5 thinking)
            max_tokens: Maximum response tokens
            stream: Stream response chunks
            thinking: Enable K2.5 thinking mode (None = auto based on temperature)
        """
        if model == "nvidia":
            try:
                return self._nvidia_completion(
                    messages, temperature, max_tokens, stream, thinking, **kwargs
                )
            except Exception as e:
                logger.warning(f"NVIDIA NIM failed ({e}), falling back to Groq")
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        reraise=True,
    )
    def _nvidia_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        thinking: Optional[bool] = None,
        **kwargs,
    ):
        """Generate completion using NVIDIA NIM Kimi K2.5."""
        self._throttle()

        # Determine thinking mode: explicit flag > auto (thinking ON for high temp)
        enable_thinking = thinking if thinking is not None else (temperature >= 0.8)

        extra_body = {
            "chat_template_kwargs": {"thinking": enable_thinking},
        }

        # K2.5 recommended params: temp=1.0/top_p=1.0 for thinking, lower for fast
        effective_temp = temperature if not enable_thinking else max(temperature, 0.9)
        effective_top_p = 1.0 if enable_thinking else 0.9

        completion = self.nvidia_client.chat.completions.create(
            model=self.nvidia_model,
            messages=messages,
            temperature=effective_temp,
            top_p=effective_top_p,
            max_tokens=max_tokens,
            stream=stream,
            extra_body=extra_body,
            **kwargs,
        )

        if stream:
            return self._stream_nvidia_response(completion)
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

    def _stream_nvidia_response(self, completion) -> Generator[str, None, None]:
        """Stream NVIDIA NIM response, yielding content tokens."""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            # Skip reasoning/thinking tokens — only yield final answer content
            if delta.content:
                yield delta.content

    def _stream_groq_response(self, completion) -> Generator[str, None, None]:
        """Stream Groq response."""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── Embeddings with Batching & Throttling ────────────────────────

    def get_embeddings(
        self,
        texts: List[str],
        input_type: str = "query",
    ) -> List[List[float]]:
        """
        Get embeddings with automatic batching and rate-limit throttling.

        Args:
            texts: Texts to embed
            input_type: "query" for search queries, "passage" for documents

        Returns:
            List of 2048-dim embedding vectors
        """
        if not texts:
            return []

        batch_size = self.settings.embedding_batch_size
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch, input_type)
            all_embeddings.extend(batch_embeddings)

            # Throttle between batches to stay within RPM limit
            if i + batch_size < len(texts):
                time.sleep(self._batch_delay)

        return all_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        reraise=True,
    )
    def _embed_batch(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        """Embed a single batch with retry logic."""
        self._throttle()

        response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            encoding_format="float",
            extra_body={
                "input_type": input_type,
                "truncate": "NONE",
            },
        )
        return [item.embedding for item in response.data]

    # ── Health Check ─────────────────────────────────────────────────

    def health_check(self) -> Dict[str, bool]:
        """Check connectivity to LLM providers."""
        health = {"nvidia": False, "groq": False, "embedding": False}

        try:
            self.nvidia_client.models.list()
            health["nvidia"] = True
        except Exception:
            pass

        if self.groq_client:
            try:
                self.groq_client.models.list()
                health["groq"] = True
            except Exception:
                pass

        try:
            self._embed_batch(["health check"], "query")
            health["embedding"] = True
        except Exception:
            pass

        return health
