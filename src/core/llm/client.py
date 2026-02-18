"""
LLM Client: Unified interface for NVIDIA NIM and Groq models.
Uses OpenAI-compatible API for NVIDIA NIM.
"""

from typing import AsyncGenerator, Dict, Any, Optional, List
import openai
from groq import Groq

from src.core.config import get_settings


class LLMClient:
    """
    Unified LLM client supporting NVIDIA NIM (primary) and Groq (secondary).
    """

    def __init__(self):
        self.settings = get_settings()

        # Initialize NVIDIA NIM client for chat (primary)
        self.nvidia_client = openai.OpenAI(
            base_url=self.settings.nvidia_base_url, api_key=self.settings.nvidia_api_key
        )
        self.nvidia_model = self.settings.nvidia_model

        # Initialize separate NVIDIA NIM client for embeddings
        self.embedding_client = openai.OpenAI(
            base_url=self.settings.nvidia_base_url,
            api_key=self.settings.nvidia_embedding_api_key
            or self.settings.nvidia_api_key,
        )
        self.embedding_model = "nvidia/llama-3.2-nv-embedqa-1b-v2"

        # Initialize Groq client (secondary) if key available
        self.groq_client = None
        if self.settings.groq_api_key:
            self.groq_client = Groq(api_key=self.settings.groq_api_key)
            self.groq_model = self.settings.groq_model

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "nvidia",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts with role and content
            model: "nvidia" or "groq"
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            **kwargs: Additional parameters

        Returns:
            Completion response or async generator if streaming
        """
        if model == "nvidia":
            return self._nvidia_completion(
                messages, temperature, max_tokens, stream, **kwargs
            )
        elif model == "groq":
            return self._groq_completion(
                messages, temperature, max_tokens, stream, **kwargs
            )
        else:
            raise ValueError(f"Unknown model: {model}")

    def _nvidia_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        **kwargs,
    ):
        """Generate completion using NVIDIA NIM."""
        completion = self.nvidia_client.chat.completions.create(
            model=self.nvidia_model,
            messages=messages,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stream=stream,
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
        """Generate completion using Groq."""
        if not self.groq_client:
            raise ValueError("Groq client not initialized. Set GROQ_API_KEY.")

        completion = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        if stream:
            return self._stream_groq_response(completion)
        else:
            return completion.choices[0].message.content

    def _stream_nvidia_response(self, completion) -> AsyncGenerator[str, None]:
        """Stream NVIDIA NIM response with reasoning content support."""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue

            # Check for reasoning content (thinking mode)
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning:
                yield f"<thinking>{reasoning}</thinking>"

            # Regular content
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _stream_groq_response(self, completion) -> AsyncGenerator[str, None]:
        """Stream Groq response."""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_embeddings(
        self, texts: List[str], model: str = "nvidia"
    ) -> List[List[float]]:
        """
        Get embeddings for texts.

        Args:
            texts: List of texts to embed
            model: "nvidia" for now (Groq doesn't support embeddings yet)

        Returns:
            List of embedding vectors
        """
        if model == "nvidia":
            # Use NVIDIA's embedding model via OpenAI-compatible API
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"},
            )
            return [item.embedding for item in response.data]
        else:
            raise ValueError(f"Embedding model {model} not supported")

    def health_check(self) -> Dict[str, bool]:
        """Check health of LLM clients."""
        health = {"nvidia": False, "groq": False}

        # Check NVIDIA
        try:
            self.nvidia_client.models.list()
            health["nvidia"] = True
        except Exception:
            pass

        # Check Groq
        if self.groq_client:
            try:
                self.groq_client.models.list()
                health["groq"] = True
            except Exception:
                pass

        return health
