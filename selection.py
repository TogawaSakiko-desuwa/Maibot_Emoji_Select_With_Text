"""Pure selection data transformations shared by the plugin runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


CONTEXT_CHARACTER_BUDGET = 4_000
CONTEXT_TRUNCATION_MARKER = "[上下文已截断，仅保留最新内容]"


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """A bounded, chronologically ordered conversation window."""

    text: str
    message_count: int
    character_count: int
    truncated: bool


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
