"""Pure selection data transformations shared by the plugin runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


CONTEXT_CHARACTER_BUDGET = 4_000
CONTEXT_TRUNCATION_MARKER = "[上下文已截断，仅保留最新内容]"
LARGE_CANDIDATE_THRESHOLD = 100


@dataclass(frozen=True, slots=True)
class EmojiCandidate:
    """One sendable emoji record from the current Host snapshot."""

    description: str
    base64_data: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Cleaned candidates plus non-fatal warning codes."""

    candidates: tuple[EmojiCandidate, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """A bounded, chronologically ordered conversation window."""

    text: str
    message_count: int
    character_count: int
    truncated: bool


@dataclass(slots=True)
class SelectionDiagnostics:
    """Non-sensitive execution facts for one tool invocation."""

    stage: str = "validate_input"
    method: str = "none"
    candidate_count: int = 0
    context_message_count: int = 0
    context_character_count: int = 0
    warnings: list[str] = field(default_factory=list)
    _started_at: float = field(default_factory=time.perf_counter, repr=False)

    def add_warning(self, code: str) -> None:
        if code not in self.warnings:
            self.warnings.append(code)

    def set_context(self, context: ContextWindow) -> None:
        self.context_message_count = context.message_count
        self.context_character_count = context.character_count
        if context.truncated:
            self.add_warning("context_truncated")

    def as_dict(self) -> dict[str, object]:
        duration_ms = max(0, round((time.perf_counter() - self._started_at) * 1_000))
        return {
            "stage": self.stage,
            "method": self.method,
            "candidate_count": self.candidate_count,
            "context_message_count": self.context_message_count,
            "context_character_count": self.context_character_count,
            "duration_ms": duration_ms,
            "warnings": list(self.warnings),
        }


def prepare_candidates(records: list[object], *, limit: int) -> CandidateSet:
    """Clean, de-duplicate and optionally limit one Host emoji snapshot."""

    if limit < 0:
        raise ValueError("limit must not be negative")

    candidates: list[EmojiCandidate] = []
    seen_descriptions: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_description = record.get("description")
        raw_base64 = record.get("base64")
        if not isinstance(raw_description, str) or not isinstance(raw_base64, str):
            continue
        description = raw_description.strip()
        base64_data = raw_base64.strip()
        if not description or not base64_data or description in seen_descriptions:
            continue

        seen_descriptions.add(description)
        candidates.append(EmojiCandidate(description, base64_data))
        if limit > 0 and len(candidates) >= limit:
            break

    warnings = (
        ("large_candidate_set",)
        if len(candidates) > LARGE_CANDIDATE_THRESHOLD
        else ()
    )
    return CandidateSet(tuple(candidates), warnings)


def format_message_block(message: object) -> str | None:
    """Format one Host message as a planner-style block."""

    if not isinstance(message, dict):
        return None

    message_info = message.get("message_info")
    if not isinstance(message_info, dict):
        message_info = {}
    user_info = message_info.get("user_info")
    if not isinstance(user_info, dict):
        user_info = {}

    content = (
        message.get("processed_plain_text")
        or message.get("plain_text")
        or message.get("content")
        or message.get("text")
        or ""
    )
    if not str(content).strip():
        return None

    timestamp = message.get("timestamp", "")
    try:
        time_text = datetime.fromtimestamp(float(timestamp)).strftime("%H:%M:%S")
    except (ValueError, TypeError, OSError, OverflowError):
        time_text = str(timestamp)

    message_id = message.get("message_id", "")
    user_name = user_info.get("user_nickname") or user_info.get("user_name") or ""
    user_card = user_info.get("user_cardname") or user_info.get("user_card") or ""

    lines: list[str] = []
    if message_id:
        lines.append(f"[msg_id]{message_id}")
    lines.extend((f"[时间]{time_text}", f"[用户名]{user_name}"))
    if user_card and user_card != user_name:
        lines.append(f"[用户群昵称]{user_card}")
    lines.append(f"[发言内容]{content}")
    return "\n".join(lines)


def build_recent_context(
    messages: list[dict[str, Any]] | list[object],
    *,
    max_characters: int = CONTEXT_CHARACTER_BUDGET,
) -> ContextWindow:
    """Keep the latest contiguous message blocks within a character budget."""

    if max_characters < 1:
        raise ValueError("max_characters must be positive")

    blocks = [block for message in messages if (block := format_message_block(message))]
    if not blocks:
        return ContextWindow("", 0, 0, False)

    newest = blocks[-1]
    if len(newest) > max_characters:
        prefix = f"{CONTEXT_TRUNCATION_MARKER}\n"
        if len(prefix) >= max_characters:
            text = prefix[:max_characters]
        else:
            text = prefix + newest[-(max_characters - len(prefix)):]
        return ContextWindow(text, 1, len(text), True)

    selected_newest_first: list[str] = []
    used_characters = 0
    for block in reversed(blocks):
        separator_size = 2 if selected_newest_first else 0
        if used_characters + separator_size + len(block) > max_characters:
            break
        selected_newest_first.append(block)
        used_characters += separator_size + len(block)

    selected = list(reversed(selected_newest_first))
    text = "\n\n".join(selected)
    return ContextWindow(
        text=text,
        message_count=len(selected),
        character_count=len(text),
        truncated=len(selected) < len(blocks),
    )
