"""
Memory System: Persistent user and company memory.
Implements selective memory with confidence scoring.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.core.config import get_settings
from src.core.llm.client import LLMClient


class MemoryEntry:
    """Represents a single memory entry."""

    def __init__(
        self,
        summary: str,
        target: str,  # "USER" or "COMPANY"
        confidence: float,
        timestamp: Optional[str] = None,
        context: Optional[str] = None,
    ):
        self.summary = summary
        self.target = target
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now().isoformat()
        self.context = context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "target": self.target,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "context": self.context,
        }


class MemoryManager:
    """
    Manages persistent memory for users and company.
    Uses selective writing with confidence scoring.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.settings = get_settings()
        self.llm_client = llm_client or LLMClient()

        # Memory file paths
        self.user_memory_path = Path("USER_MEMORY.md")
        self.company_memory_path = Path("COMPANY_MEMORY.md")

    def should_write_memory(
        self, conversation: str, current_user_memory: str, current_company_memory: str
    ) -> Dict[str, Any]:
        """
        Determine if memory should be written based on conversation.

        Args:
            conversation: Recent conversation text
            current_user_memory: Existing user memory
            current_company_memory: Existing company memory

        Returns:
            Decision dict with should_write, target, summary, confidence
        """
        prompt = f"""Analyze this conversation and determine if there's valuable information to remember.

Current User Memory:
{current_user_memory if current_user_memory else "(empty)"}

Current Company Memory:
{current_company_memory if current_company_memory else "(empty)"}

Recent Conversation:
{conversation}

Instructions:
1. Identify high-value facts (preferences, role, workflows, patterns)
2. Avoid: casual conversation, transient topics, already-known info
3. Choose target: USER (personal) or COMPANY (organizational)
4. Assign confidence (0.0-1.0) based on clarity and importance
5. If no valuable info, set should_write to false

Respond in this exact JSON format:
{{
    "should_write": true/false,
    "target": "USER" or "COMPANY" or "NONE",
    "summary": "concise fact to remember",
    "confidence": 0.0-1.0
}}"""

        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory management system. Be selective and only record high-value information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.memory_decision_temperature,
                max_tokens=500,
            )

            # Parse JSON response
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                decision = json.loads(response)

            # Validate decision
            if (
                decision.get("confidence", 0)
                < self.settings.memory_confidence_threshold
            ):
                decision["should_write"] = False

            return decision

        except Exception as e:
            print(f"Error in memory decision: {e}")
            return {
                "should_write": False,
                "target": "NONE",
                "summary": "",
                "confidence": 0.0,
            }

    def write_memory(self, entry: MemoryEntry) -> bool:
        """
        Write memory entry to appropriate file.

        Args:
            entry: Memory entry to write

        Returns:
            True if successful
        """
        # Check for duplicates
        if self._is_duplicate(entry):
            print(f"Skipping duplicate memory: {entry.summary[:50]}...")
            return False

        # Format entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry_text = (
            f"\n- [{timestamp}] {entry.summary} (confidence: {entry.confidence:.2f})"
        )

        # Write to appropriate file
        if entry.target == "USER":
            with open(self.user_memory_path, "a", encoding="utf-8") as f:
                f.write(entry_text)
            print(f"Written to USER_MEMORY.md: {entry.summary[:50]}...")
        elif entry.target == "COMPANY":
            with open(self.company_memory_path, "a", encoding="utf-8") as f:
                f.write(entry_text)
            print(f"Written to COMPANY_MEMORY.md: {entry.summary[:50]}...")
        else:
            return False

        return True

    def _is_duplicate(self, new_entry: MemoryEntry) -> bool:
        """Check if entry is duplicate of existing memory."""
        target_file = (
            self.user_memory_path
            if new_entry.target == "USER"
            else self.company_memory_path
        )

        if not target_file.exists():
            return False

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple check: does summary text already exist?
        return new_entry.summary.lower() in content.lower()

    def read_memory(self, target: str) -> str:
        """
        Read memory from file.

        Args:
            target: "USER" or "COMPANY"

        Returns:
            Memory content
        """
        target_file = (
            self.user_memory_path if target == "USER" else self.company_memory_path
        )

        if target_file.exists():
            with open(target_file, "r", encoding="utf-8") as f:
                return f.read()

        return ""

    def get_relevant_memories(
        self, query: str, target: str = "USER", k: int = 3
    ) -> List[str]:
        """
        Get memories relevant to query.
        Simple keyword matching for now.

        Args:
            query: Query to match against memories
            target: "USER" or "COMPANY"
            k: Number of memories to return

        Returns:
            List of relevant memory strings
        """
        memory_content = self.read_memory(target)

        if not memory_content:
            return []

        # Split into individual memories
        memories = [m.strip() for m in memory_content.split("\n-") if m.strip()]

        # Simple keyword matching
        query_words = set(query.lower().split())
        scored_memories = []

        for memory in memories:
            memory_words = set(memory.lower().split())
            overlap = len(query_words & memory_words)
            scored_memories.append((memory, overlap))

        # Sort by relevance and return top k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored_memories[:k]]

    def update_memory_from_conversation(
        self, conversation: str
    ) -> Optional[MemoryEntry]:
        """
        Main entry point: analyze conversation and update memory if needed.

        Args:
            conversation: Recent conversation

        Returns:
            MemoryEntry if written, None otherwise
        """
        user_memory = self.read_memory("USER")
        company_memory = self.read_memory("COMPANY")

        decision = self.should_write_memory(conversation, user_memory, company_memory)

        if decision.get("should_write") and decision.get("target") in [
            "USER",
            "COMPANY",
        ]:
            entry = MemoryEntry(
                summary=decision["summary"],
                target=decision["target"],
                confidence=decision["confidence"],
                context=conversation[:500],  # First 500 chars as context
            )

            if self.write_memory(entry):
                return entry

        return None
