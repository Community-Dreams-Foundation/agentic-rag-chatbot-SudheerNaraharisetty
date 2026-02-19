"""
Memory System: Persistent user and company memory with selective writing.
Uses LLM-based decision making with confidence scoring.
Thinking mode is OFF for memory decisions to save tokens and latency.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.core.config import get_settings
from src.core.llm.client import LLMClient

logger = logging.getLogger(__name__)


class MemoryEntry:
    """Represents a single memory entry with metadata."""

    __slots__ = ("summary", "target", "confidence", "timestamp", "context")

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
    Manages persistent memory for users and company knowledge.

    Key design decisions:
    - Uses LLM with thinking=False and low temperature for fast, deterministic decisions
    - Selective writing: only stores high-signal, reusable facts
    - Duplicate detection prevents redundant entries
    - Markdown format for human readability and hackathon requirements
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.settings = get_settings()
        self.llm_client = llm_client or LLMClient()

        # Memory file paths — in project root as required by hackathon
        self.user_memory_path = Path("USER_MEMORY.md")
        self.company_memory_path = Path("COMPANY_MEMORY.md")

        # Ensure files exist
        self._ensure_memory_files()

    def _ensure_memory_files(self):
        """Create memory files with headers if they don't exist."""
        if not self.user_memory_path.exists():
            self.user_memory_path.write_text(
                "# User Memory\n\nSelective, high-signal facts about the user.\n\n",
                encoding="utf-8",
            )
        if not self.company_memory_path.exists():
            self.company_memory_path.write_text(
                "# Company Memory\n\nOrganizational learnings and reusable knowledge.\n\n",
                encoding="utf-8",
            )

    # ── Memory Decision ─────────────────────────────────────────────

    def should_write_memory(
        self, conversation: str, current_user_memory: str, current_company_memory: str
    ) -> Dict[str, Any]:
        """
        Use LLM to decide if conversation contains memory-worthy information.

        Uses low temperature (0.3) and thinking=False for fast, deterministic
        decisions that don't waste tokens on reasoning.

        Returns:
            Decision dict: {should_write, target, summary, confidence}
        """
        prompt = f"""Analyze this conversation and determine if it contains valuable information worth remembering long-term.

Current User Memory:
{current_user_memory[:500] if current_user_memory else "(empty)"}

Current Company Memory:
{current_company_memory[:500] if current_company_memory else "(empty)"}

Recent Conversation:
{conversation[:1000]}

Decision criteria:
- WRITE if: user role/title, explicit preferences, workflow patterns, recurring topics, organizational structure, domain expertise
- SKIP if: casual chat, one-time questions, already known facts, transient topics, greetings
- Target USER for personal facts (role, preferences, habits)
- Target COMPANY for organizational knowledge (processes, tools, team structure)
- Confidence 0.0-1.0: how certain and how reusable is this fact?

Respond with ONLY this JSON (no other text):
{{"should_write": true, "target": "USER", "summary": "concise fact", "confidence": 0.85}}

Or if nothing worth remembering:
{{"should_write": false, "target": "NONE", "summary": "", "confidence": 0.0}}"""

        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a memory management system. Output ONLY valid JSON. "
                            "Be highly selective — only record information that would be "
                            "useful in future conversations."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.memory_decision_temperature,
                max_tokens=300,
                thinking=False,  # Explicit: no thinking tokens for memory decisions
            )

            if not response:
                return self._default_no_write()

            # Parse JSON from response — handle LLM wrapping it in text
            decision = self._parse_json_response(response)

            # Enforce confidence threshold
            if (
                decision.get("confidence", 0)
                < self.settings.memory_confidence_threshold
            ):
                decision["should_write"] = False

            return decision

        except Exception as e:
            logger.error(f"Memory decision failed: {e}")
            return self._default_no_write()

    @staticmethod
    def _parse_json_response(response: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response, handling markdown code blocks
        and extra text around the JSON.
        """
        # Try direct parse first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding a JSON object with nested content (handles multiline)
        json_match = re.search(r"\{[^{}]*\"should_write\"[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"should_write": False, "target": "NONE", "summary": "", "confidence": 0.0}

    @staticmethod
    def _default_no_write() -> Dict[str, Any]:
        return {
            "should_write": False,
            "target": "NONE",
            "summary": "",
            "confidence": 0.0,
        }

    # ── Memory Writing ──────────────────────────────────────────────

    def write_memory(self, entry: MemoryEntry) -> bool:
        """
        Write a memory entry to the appropriate markdown file.

        Returns True if written, False if skipped (duplicate or invalid target).
        """
        if entry.target not in ("USER", "COMPANY"):
            return False

        # Duplicate check
        if self._is_duplicate(entry):
            logger.info(f"Skipping duplicate memory: {entry.summary[:50]}...")
            return False

        # Format entry as markdown list item
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry_text = (
            f"\n- [{timestamp}] {entry.summary} "
            f"(confidence: {entry.confidence:.2f})"
        )

        target_file = (
            self.user_memory_path
            if entry.target == "USER"
            else self.company_memory_path
        )

        with open(target_file, "a", encoding="utf-8") as f:
            f.write(entry_text)

        logger.info(f"Memory written to {target_file.name}: {entry.summary[:50]}...")
        return True

    def _is_duplicate(self, new_entry: MemoryEntry) -> bool:
        """Check if an entry already exists in the target memory file."""
        target_file = (
            self.user_memory_path
            if new_entry.target == "USER"
            else self.company_memory_path
        )

        if not target_file.exists():
            return False

        content = target_file.read_text(encoding="utf-8").lower()
        return new_entry.summary.lower() in content

    # ── Memory Reading ──────────────────────────────────────────────

    def read_memory(self, target: str) -> str:
        """Read memory content from file."""
        target_file = (
            self.user_memory_path if target == "USER" else self.company_memory_path
        )

        if target_file.exists():
            return target_file.read_text(encoding="utf-8")

        return ""

    def get_relevant_memories(
        self, query: str, target: str = "USER", k: int = 3
    ) -> List[str]:
        """
        Retrieve memories relevant to a query using keyword overlap scoring.

        Args:
            query: Search query to match against
            target: "USER" or "COMPANY"
            k: Maximum number of memories to return

        Returns:
            List of relevant memory strings, sorted by relevance
        """
        memory_content = self.read_memory(target)

        if not memory_content:
            return []

        # Split into individual memory entries
        memories = [m.strip() for m in memory_content.split("\n-") if m.strip()]

        # Score by keyword overlap
        query_words = set(query.lower().split())
        scored = []

        for memory in memories:
            memory_words = set(memory.lower().split())
            overlap = len(query_words & memory_words)
            if overlap > 0:
                scored.append((memory, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:k]]

    def get_all_memories_for_context(self) -> str:
        """
        Get a combined summary of all memories for injection into agent context.
        Used to personalize responses.
        """
        user_mem = self.read_memory("USER")
        company_mem = self.read_memory("COMPANY")

        parts = []
        if user_mem and len(user_mem.strip()) > 50:  # Skip if just header
            parts.append(f"User context:\n{user_mem[:500]}")
        if company_mem and len(company_mem.strip()) > 50:
            parts.append(f"Company context:\n{company_mem[:500]}")

        return "\n\n".join(parts) if parts else ""

    # ── Main Entry Point ────────────────────────────────────────────

    def update_memory_from_conversation(
        self, conversation: str
    ) -> Optional[MemoryEntry]:
        """
        Analyze conversation and update memory if warranted.

        This is the main entry point called after each interaction.

        Args:
            conversation: Recent conversation text (user + assistant)

        Returns:
            MemoryEntry if written, None otherwise
        """
        user_memory = self.read_memory("USER")
        company_memory = self.read_memory("COMPANY")

        decision = self.should_write_memory(conversation, user_memory, company_memory)

        if decision.get("should_write") and decision.get("target") in (
            "USER",
            "COMPANY",
        ):
            entry = MemoryEntry(
                summary=decision["summary"],
                target=decision["target"],
                confidence=decision.get("confidence", 0.7),
                context=conversation[:500],
            )

            if self.write_memory(entry):
                return entry

        return None

    # ── Agent Tool Interface ────────────────────────────────────────

    def write_memory_from_agent(
        self, target: str, summary: str
    ) -> Dict[str, Any]:
        """
        Write memory directly from the agent tool.

        Called by the agent as:
          {"tool": "write_memory", "args": {"target": "USER", "summary": "..."}}

        Returns:
            Result dict for agent observation
        """
        if target not in ("USER", "COMPANY"):
            return {"success": False, "error": f"Invalid target: {target}"}

        entry = MemoryEntry(
            summary=summary,
            target=target,
            confidence=0.9,  # Agent-driven writes have high confidence
        )

        written = self.write_memory(entry)
        return {
            "success": written,
            "target": target,
            "summary": summary,
            "message": "Memory saved" if written else "Duplicate — already known",
        }
