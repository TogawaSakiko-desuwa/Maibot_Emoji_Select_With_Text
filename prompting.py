"""Pure prompt construction and LLM response parsing."""

from pathlib import Path
from typing import Any

import json
import logging
import re

from . import selection


logger = logging.getLogger("Maibot_Emoji_Select_With_Text")
_SELECTION_PROMPT_PATH = Path(__file__).parent / "select_emoji.prompt"
_FALLBACK_SELECTION_PROMPT = """\
阅读以下对话上下文和当前想表达的情感，从{emoji_count}个表情包描述中选择最匹配的一个：

{conversation_context}
{emotion_hint_block}
{description_list}

仅返回JSON：{{"selected": 1}}（单个编号）"""


def load_prompt_template() -> str:
    """加载提示词模板文件。文件不存在时返回内置默认值。"""

    try:
        return _SELECTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return _FALLBACK_SELECTION_PROMPT


def build_conversation_context(messages: list[dict[str, Any]]) -> str:
    """将原始消息列表构建为接近 planner 格式的对话上下文。"""

    blocks = [
        block
        for message in messages
        if (block := selection.format_message_block(message))
    ]
    return "\n\n".join(blocks)


def build_selection_prompt(
    descriptions: list[str],
    conversation_context: str = "",
    emotion_expression: str = "",
) -> str:
    """构建发给文本 LLM 的表情包选择 prompt。"""

    values = {
        "emoji_count": str(len(descriptions)),
        "conversation_context": conversation_context,
        "emotion_hint_block": f"当前想表达的情感：{emotion_expression}" if emotion_expression else "",
        "description_list": "\n".join(f"{index + 1}. {description}" for index, description in enumerate(descriptions)),
    }
    try:
        return load_prompt_template().format(**values)
    except (KeyError, ValueError):
        logger.warning("[EmojiTextSelector] 自定义 prompt 模板格式无效，回退到内置模板")
        return _FALLBACK_SELECTION_PROMPT.format(**values)


def parse_llm_index(response_text: str, max_count: int) -> int | None:
    """从 LLM 返回文本的全部 JSON 候选中解析单个编号。"""

    text = (response_text or "").strip()
    if not text or max_count < 1:
        return None
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in re.finditer(r"```json\s*(.*?)```", text, re.DOTALL))
    candidates.extend(match.group(1).strip() for match in re.finditer(r"```\s*(.*?)```", text, re.DOTALL))
    candidates.extend(match.group(0).strip() for match in re.finditer(r"\{[^{}]*\}", text))
    for candidate in candidates:
        selected = _try_parse_index(candidate, max_count)
        if selected is not None:
            return selected
    return None


def _try_parse_index(candidate: str, max_count: int) -> int | None:
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return None
    if not isinstance(data, dict):
        return None
    selected = data.get("selected")
    if type(selected) is not int:
        return None
    return selected if 1 <= selected <= max_count else None
