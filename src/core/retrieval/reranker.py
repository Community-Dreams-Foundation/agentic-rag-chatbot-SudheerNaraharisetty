"""
NVIDIA NIM Reranker: Cross-encoder reranking using llama-3.2-nv-rerankqa-1b-v2.
Designed to work with llama-3.2-nv-embedqa-1b-v2 embeddings for optimal retrieval.

Key features:
- 8192 token context (16x more than Mistral-4B reranker)
- 26 language support + cross-lingual retrieval
- 3.5x smaller than Mistral-4B, lower latency
- Perfect pair with our embedding model
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class RerankResult:
    """Container for a single reranked passage."""

    __slots__ = ("index", "logit", "text", "metadata", "probability")

    def __init__(
        self,
        index: int,
        logit: float,
        text: str,
        metadata: Optional[Dict] = None,
    ):
        self.index = index
        self.logit = logit
        self.text = text
        self.metadata = metadata or {}
        self.probability = self._sigmoid(logit)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Convert logit to probability (0-1 range)."""
        import math

        return 1 / (1 + math.exp(-x))

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "logit": self.logit,
            "probability": round(self.probability, 4),
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
        }


class NVIDIAReranker:
    """
    NVIDIA NIM Reranker using llama-3.2-nv-rerankqa-1b-v2.

    This reranker is specifically designed to work with the
    llama-3.2-nv-embedqa-1b-v2 embedding model we use, providing
    optimal retrieval accuracy as a unified pipeline.

    Architecture:
        Query + Candidate Passages → Cross-Encoder → Relevance Scores

    Benchmarks (with embedqa-1b-v2):
        - NQ + HotpotQA + FiQA + TechQA: 73.64% Recall@5
        - MIRACL Multilingual: 65.80% Recall@5
        - Cross-lingual (MLQA): 86.83% Recall@5
        - Long Documents (MLDR): 70.69% Recall@5
    """

    def __init__(self):
        self.settings = get_settings()
        self.api_key = (
            self.settings.nvidia_rerank_api_key or self.settings.nvidia_api_key
        )
        self.model = self.settings.rerank_model
        self.base_url = self.settings.nvidia_rerank_base_url

        self._request_timestamps: List[float] = []
        self._rpm_limit = self.settings.api_requests_per_minute

        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        """Lazy-load requests session for connection reuse."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

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
                logger.info(f"Reranker rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._request_timestamps.append(time.time())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (
                requests.exceptions.RequestException,
                requests.exceptions.HTTPError,
            )
        ),
        reraise=True,
    )
    def rerank(
        self,
        query: str,
        passages: List[Dict],
        top_k: int = 5,
        truncate: str = "END",
    ) -> List[RerankResult]:
        """
        Rerank passages by relevance to query using NVIDIA NIM cross-encoder.

        Args:
            query: The search query
            passages: List of dicts with 'text' key (and optional 'metadata')
            top_k: Number of top results to return
            truncate: Truncation strategy ('END' or 'NONE')

        Returns:
            List of RerankResult objects sorted by relevance (descending)
        """
        if not passages:
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided to reranker")
            return [
                RerankResult(i, 0.0, p.get("text", ""), p.get("metadata"))
                for i, p in enumerate(passages[:top_k])
            ]

        # Filter out empty passages — NVIDIA API returns 400 on empty text
        valid_indices = []
        valid_passages = []
        for i, p in enumerate(passages):
            text = (p.get("text") or "").strip()
            if text:
                valid_indices.append(i)
                valid_passages.append(p)

        if not valid_passages:
            logger.warning("All passages empty after filtering, skipping rerank")
            return []

        self._throttle()

        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": p.get("text", "")} for p in valid_passages],
            "truncate": truncate,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        session = self._get_session()

        try:
            response = session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.HTTPError as e:
            status = (
                getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None
            )
            if status == 429:
                logger.warning("Reranker rate limited (429), retrying...")
            logger.error(f"Reranker HTTP error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Reranker request failed: {e}")
            raise

        rankings = data.get("rankings", [])
        if not rankings:
            logger.warning("Reranker returned empty rankings")
            return []

        results = []
        for ranking in rankings[:top_k]:
            filtered_idx = ranking.get("index", 0)
            logit = ranking.get("logit", 0.0)
            # Map back to original passage index
            original_idx = (
                valid_indices[filtered_idx]
                if filtered_idx < len(valid_indices)
                else filtered_idx
            )
            passage = passages[original_idx] if original_idx < len(passages) else {}
            results.append(
                RerankResult(
                    index=original_idx,
                    logit=logit,
                    text=passage.get("text", ""),
                    metadata=passage.get("metadata"),
                )
            )

        logger.info(
            f"Reranked {len(passages)} passages, top result logit={results[0].logit:.2f}"
            if results
            else "No rerank results"
        )

        return results

    def rerank_texts(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float, str]]:
        """
        Convenience method for reranking plain text list.

        Args:
            query: The search query
            texts: List of text strings to rerank
            top_k: Number of top results to return

        Returns:
            List of (original_index, logit, text) tuples
        """
        passages = [{"text": t} for t in texts]
        results = self.rerank(query, passages, top_k=top_k)
        return [(r.index, r.logit, r.text) for r in results]

    def health_check(self) -> Dict[str, bool]:
        """Test reranker connectivity."""
        try:
            result = self.rerank("test", [{"text": "test passage"}], top_k=1)
            return {"reranker": len(result) > 0}
        except Exception as e:
            logger.error(f"Reranker health check failed: {e}")
            return {"reranker": False}


_reranker_instance: Optional[NVIDIAReranker] = None


def get_reranker() -> NVIDIAReranker:
    """Get singleton reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = NVIDIAReranker()
    return _reranker_instance
