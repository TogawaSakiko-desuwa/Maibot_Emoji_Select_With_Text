"""基于表情包描述文本的表情包选择插件。

通过 ``@Tool("select_emoji_with_text", core_tool=True)`` 注册为直接可见的核心工具。
1. 一次性获取带图片数据的表情包快照
2. 清洗并按 description 去重后建立稳定编号
3. 使用语义匹配或文本 LLM 选择一个编号
4. 从同一快照发送选中记录，失败时返回明确错误且不随机兜底
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from maibot_sdk import HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import HookMode, ToolParameterInfo, ToolParamType

from . import config_models, emoji_cache, prompting, selection

logger = logging.getLogger("Maibot_Emoji_Select_With_Text")
# ─── 插件主类 ───────────────────────────────────────────────────


class EmojiTextSelectorPlugin(MaiBotPlugin):
    """表情包选择插件，支持语义向量匹配 + 文本 LLM 选择两级策略。"""

    config_model: ClassVar[type[PluginConfigBase] | None] = config_models.EmojiTextSelectorConfig

    def __init__(self) -> None:
        super().__init__()
        self._cache = emoji_cache.EmojiEmbeddingCache()
        self._refresh_task: Optional[asyncio.Task] = None
        self._plugin_dir: Optional[Path] = None

    @classmethod
    def build_config_schema(
        cls,
        *,
        plugin_id: str = "",
        plugin_name: str = "",
        plugin_version: str = "",
        plugin_description: str = "",
        plugin_author: str = "",
    ) -> dict[str, Any]:
        schema = super().build_config_schema(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            plugin_description=plugin_description,
            plugin_author=plugin_author,
        )
        # 隐藏 [plugin] 节（含 config_version），普通用户无需关心
        schema.get("sections", {}).pop("plugin", None)
        return schema

    # ─── 生命周期 ────────────────────────────────────────────

    async def on_load(self) -> None:
        self._plugin_dir = Path(__file__).parent
        cache_dir = self._plugin_dir / ".cache"

        if self.config.semantic.enabled:
            if self._cache.load_from_disk(cache_dir):
                logger.info(f"[EmojiTextSelector] 从磁盘恢复向量缓存成功，共 {self._cache.count} 条")
            self._start_refresh_task()
        logger.info("[EmojiTextSelector] 插件已加载")

    async def on_unload(self) -> None:
        await self._stop_refresh_task()

        if self._plugin_dir is not None:
            self._cache.save_to_disk(self._plugin_dir / ".cache")

        logger.info("[EmojiTextSelector] 插件已卸载")

    async def on_config_update(
        self, scope: str, config_data: dict[str, Any], version: str
    ) -> None:
        del config_data, version
        if scope != "self":
            return
        if self.config.semantic.enabled:
            if self._plugin_dir is not None and self._cache.is_empty:
                self._cache.load_from_disk(self._plugin_dir / ".cache")
            self._start_refresh_task()
        else:
            await self._stop_refresh_task()

    def _start_refresh_task(self) -> None:
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._background_refresh_loop())

    async def _stop_refresh_task(self) -> None:
        task = self._refresh_task
        self._refresh_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _prepare_candidates(self, records: list[object]) -> selection.CandidateSet:
        """Return cleaned candidates from one Host snapshot."""

        return selection.prepare_candidates(
            records,
            limit=self.config.selector.max_emotion_tags,
        )

    @staticmethod
    def _finish_tool_result(
        diagnostics: selection.SelectionDiagnostics,
        *,
        success: bool,
        stage: str,
        selection_method: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Build one stable tool result and emit only non-sensitive facts."""

        diagnostics.stage = stage
        if selection_method is not None:
            diagnostics.method = selection_method
        details = diagnostics.as_dict()
        logger.info(
            "[EmojiTextSelector] success=%s stage=%s method=%s candidates=%d "
            "context_messages=%d context_chars=%d duration_ms=%d warnings=%s",
            success,
            details["stage"],
            details["method"],
            details["candidate_count"],
            details["context_message_count"],
            details["context_character_count"],
            details["duration_ms"],
            ",".join(details["warnings"]) or "none",
        )
        result = {"success": success, **fields, "diagnostics": details}
        if success and selection_method is not None:
            result["method"] = selection_method
        return result

    # ─── 向量缓存刷新 ────────────────────────────────────────

    async def _background_refresh_loop(self) -> None:
        """后台定时刷新向量缓存。仅在语义匹配启用时运行。"""
        await asyncio.sleep(5)
        while True:
            try:
                if not self.config.semantic.enabled:
                    await asyncio.sleep(30)
                    continue
                interval = self.config.semantic.refresh_interval_seconds
                if self._cache.needs_refresh(interval):
                    await self._refresh_cache()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[EmojiTextSelector] 向量缓存刷新失败: {exc}")
            await asyncio.sleep(30)

    async def _refresh_cache(self) -> None:
        """从实际表情记录加载描述，增量计算 embedding 向量。

        不再按标签调用 ``get_by_description``。核心模糊检索在没有命中时
        仍可能返回零相似度的随机表情，使用其结果构建索引会污染描述与
        图片之间的对应关系。
        """
        self._cache.set_refreshing()
        refresh_start = time.time()
        try:
            try:
                emojis: list[dict[str, Any]] = await self.ctx.emoji.get_all()
            except Exception as exc:
                logger.warning(f"[EmojiTextSelector] 获取表情记录失败: {exc}")
                return

            if not isinstance(emojis, list) or not emojis:
                return

            emojis = self._prepare_candidates(emojis).candidates
            if not emojis:
                return

            # 使用真实描述作为稳定键，确保描述与图片来自同一条记录。
            old_tag_to_id: Dict[str, int] = self._cache.get_tag_to_id_map()
            next_id = max(old_tag_to_id.values()) + 1 if old_tag_to_id else 0

            # 分配稳定的 cache_id；候选已在共享入口完成清洗与去重。
            valid_ids: set[int] = set()
            changed_ids: set[int] = set()
            texts_to_embed: List[Tuple[int, str]] = []
            id_to_tag: Dict[int, str] = {}

            for candidate in emojis:
                desc = candidate.description
                tag = desc

                cache_id = old_tag_to_id.get(tag, next_id)
                if cache_id == next_id:
                    next_id += 1
                valid_ids.add(cache_id)
                id_to_tag[cache_id] = tag

                if self._cache.get_text_key(cache_id) != desc:
                    changed_ids.add(cache_id)
                    texts_to_embed.append((cache_id, desc))

            # 保留已有条目（排除变更的，其旧向量将在后面被新向量覆盖）
            kept_ids, kept_matrix, kept_text_keys, kept_emotion_tags = (
                self._cache.get_existing_entries(valid_ids, changed_ids)
            )

            # 分批计算新增/变更的 embedding
            batch_size = max(1, self.config.semantic.embed_batch_size)
            new_ids: List[int] = []
            new_vectors: List[List[float]] = []
            new_text_keys: Dict[int, str] = {}
            new_emotion_tags: Dict[int, str] = {}

            if texts_to_embed:
                for batch_start in range(0, len(texts_to_embed), batch_size):
                    batch_items = texts_to_embed[batch_start:batch_start + batch_size]
                    batch_texts = [text_key for _, text_key in batch_items]

                    embed_result = None
                    for attempt in range(2):
                        try:
                            embed_result = await self.ctx.llm.embed(texts=batch_texts)
                            break
                        except Exception as exc:
                            if attempt == 0:
                                logger.warning(
                                    f"[EmojiTextSelector] embedding 调用失败（第1次），"
                                    f"10s 后重试: {exc}"
                                )
                                await asyncio.sleep(10)
                            else:
                                logger.error(
                                    f"[EmojiTextSelector] embedding 调用失败（第2次），"
                                    f"跳过当前批次 ({len(batch_items)} 条): {exc}"
                                )

                    if embed_result is None:
                        continue

                    if isinstance(embed_result, dict) and embed_result.get("success"):
                        emb_results = embed_result.get("results", [])
                        if len(emb_results) < len(batch_items):
                            dropped = len(batch_items) - len(emb_results)
                            logger.warning(
                                f"[EmojiTextSelector] embedding API 返回结果不足: "
                                f"请求 {len(batch_items)} 条，仅收到 {len(emb_results)} 条，"
                                f"{dropped} 条描述将回退到旧缓存"
                            )
                        for i, (cache_id, text_key) in enumerate(batch_items):
                            if i < len(emb_results):
                                vector = emb_results[i].get("embedding", [])
                                if vector:
                                    new_ids.append(cache_id)
                                    new_vectors.append(vector)
                                    new_text_keys[cache_id] = text_key
                                    new_emotion_tags[cache_id] = id_to_tag.get(cache_id, "")
                    else:
                        logger.warning(f"[EmojiTextSelector] 批量 embedding 失败: {embed_result}")

            # 合并：new 覆盖 kept 中的同 id 条目（embedding 成功时用新向量，失败时保留旧向量）
            new_id_set = set(new_ids)
            final_ids: List[int] = []
            final_text_keys: Dict[int, str] = {}
            final_emotion_tags: Dict[int, str] = {}
            matrix_rows: List[np.ndarray] = []

            for i, cid in enumerate(kept_ids):
                if cid in new_id_set:
                    continue
                final_ids.append(cid)
                final_text_keys[cid] = kept_text_keys.get(cid, "")
                final_emotion_tags[cid] = kept_emotion_tags.get(cid, "")
                if kept_matrix.size > 0:
                    matrix_rows.append(kept_matrix[i])

            for i, cid in enumerate(new_ids):
                final_ids.append(cid)
                final_text_keys[cid] = new_text_keys.get(cid, "")
                final_emotion_tags[cid] = new_emotion_tags.get(cid, "")
                matrix_rows.append(np.array(new_vectors[i], dtype=np.float32))

            if matrix_rows:
                all_matrix = np.vstack(matrix_rows)
            else:
                all_matrix = np.empty((0, 0), dtype=np.float32)

            self._cache.rebuild(final_ids, final_text_keys, final_emotion_tags, all_matrix)

            if self._plugin_dir is not None:
                self._cache.save_to_disk(self._plugin_dir / ".cache")

            refresh_elapsed_ms = (time.time() - refresh_start) * 1000
            logger.info(
                f"[EmojiTextSelector] 向量缓存刷新完成，共 {self._cache.count} 条表情包描述，"
                f"本次新增/更新 {len(new_ids)} 条，耗时 {refresh_elapsed_ms:.0f}ms"
            )
        finally:
            self._cache.mark_refreshed()

    async def _semantic_select(
        self,
        query_text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """语义向量匹配。返回 (matched_tag, matched_description) 或 (None, None)。"""
        if not query_text.strip():
            return None, None

        try:
            embed_result = await self.ctx.llm.embed(text=query_text)
            if not isinstance(embed_result, dict) or not embed_result.get("success"):
                logger.warning("[EmojiTextSelector] 查询 embedding 失败")
                return None, None

            raw_vector = embed_result.get("embedding", [])
            if not raw_vector:
                return None, None

            query_vector = np.array(raw_vector, dtype=np.float32)
            q_norm = np.linalg.norm(query_vector)
            if q_norm < 1e-12:
                return None, None
            query_vector = query_vector / q_norm

            threshold = self.config.semantic.similarity_threshold
            top_matches = self._cache.search(query_vector, threshold, max_count=1)
            if not top_matches:
                logger.info("[EmojiTextSelector] 语义匹配未找到超过阈值的表情包")
                return None, None

            best_id, best_score = top_matches[0]
            matched_tag = self._cache.get_emotion_tag(best_id)
            matched_desc = self._cache.get_text_key(best_id)
            logger.info(
                "[EmojiTextSelector] 语义匹配命中: score=%.3f",
                best_score,
            )
            return matched_tag, matched_desc
        except Exception as exc:
            logger.error(
                "[EmojiTextSelector] 语义匹配异常: %s",
                type(exc).__name__,
            )
            return None, None

    async def _fetch_conversation_context(self, stream_id: str) -> selection.ContextWindow:
        """获取最近的完整消息块，并按固定字符预算保留最新上下文。"""
        if not stream_id:
            return selection.ContextWindow("", 0, 0, False)
        try:
            messages = await self.ctx.message.get_recent(
                stream_id,
                limit=self.config.selector.context_message_limit,
            )
            if not messages:
                return selection.ContextWindow("", 0, 0, False)
            return selection.build_recent_context(messages)
        except Exception as exc:
            logger.debug(
                "[EmojiTextSelector] 获取对话上下文异常: %s",
                type(exc).__name__,
            )
            return selection.ContextWindow("", 0, 0, False)

    # ─── Tool: select_emoji_with_text ────────────────────────

    @Tool(
        name="select_emoji_with_text",
        description="根据当前对话情绪从表情包库中选择并发送合适的表情包。",
        parameters=[
            ToolParameterInfo(
                name="emotion_hint",
                param_type=ToolParamType.STRING,
                description="你想通过表情包表达的情感或态度，例如：'对刚才的玩笑表示开心和赞同'、'表达无奈和吐槽'、'给对方鼓励和安慰'",
                required=False,
            ),
        ],
        core_tool=True,
        visibility="visible",
        timeout_ms=150_000,
    )
    async def handle_select_emoji(
        self,
        stream_id: str = "",
        emotion_hint: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """处理 select_emoji_with_text 工具调用。

        1. 如果启用语义匹配且缓存就绪，优先使用 embedding 向量匹配
        2. 向量匹配失败则降级为文本 LLM 选择
        3. 两级都失败则返回 error

        始终从实际表情记录建立 description→emoji 精确映射，避免核心模糊
        查询在低相似度时返回无关表情，造成“看对了、发错了”。
        """
        del kwargs
        diagnostics = selection.SelectionDiagnostics()

        if not isinstance(stream_id, str) or not stream_id.strip():
            return self._finish_tool_result(
                diagnostics,
                success=False,
                stage="validate_input",
                error="缺少消息流 ID",
            )

        try:
            # 1. 读取实际表情记录。必须先取得真实记录，再做选择；不能先用
            # get_emotions() 获得标签后逐个模糊查询，否则零相似度候选也可能
            # 被核心查询返回，污染候选列表并发送错误图片。
            try:
                raw_emoji_records = await self.ctx.emoji.get_all()
            except Exception as exc:
                logger.warning(
                    "[EmojiTextSelector] 读取表情包库失败: %s",
                    type(exc).__name__,
                )
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="read_library",
                    error="读取表情包库失败",
                )

            if isinstance(raw_emoji_records, dict):
                if raw_emoji_records.get("success") is True and isinstance(
                    raw_emoji_records.get("emojis"), list
                ):
                    raw_emoji_records = raw_emoji_records["emojis"]
                else:
                    logger.warning("[EmojiTextSelector] Host 返回表情包读取失败")
                    return self._finish_tool_result(
                        diagnostics,
                        success=False,
                        stage="read_library",
                        error="读取表情包库失败",
                    )
            elif not isinstance(raw_emoji_records, list):
                logger.warning(
                    "[EmojiTextSelector] 表情包库返回了未知数据结构: %s",
                    type(raw_emoji_records).__name__,
                )
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="read_library",
                    error="读取表情包库失败",
                )

            if not raw_emoji_records:
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="read_library",
                    error="表情包库中没有可用记录",
                )

            # 2. 以同一份已清洗候选建立稳定的精确映射。文本和语义路径
            # 共用这个集合，因此都遵守去重规则和候选数量限制。
            candidate_set = self._prepare_candidates(raw_emoji_records)
            candidates = candidate_set.candidates
            diagnostics.candidate_count = len(candidates)
            for warning in candidate_set.warnings:
                diagnostics.add_warning(warning)
            desc_to_emoji = {candidate.description: candidate for candidate in candidates}
            ordered_descriptions = list(desc_to_emoji)

            logger.debug(
                f"[EmojiTextSelector] 从 {len(raw_emoji_records)} 条实际记录读取到 "
                f"{len(ordered_descriptions)} 个唯一表情描述"
            )

            if not ordered_descriptions:
                logger.error(
                    "[EmojiTextSelector] 没有同时包含描述和图片数据的表情包记录"
                )
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="prepare_candidates",
                    error="表情包库中没有包含图片数据的可用记录",
                )

            context_window = await self._fetch_conversation_context(stream_id)
            diagnostics.set_context(context_window)
            extra_context = context_window.text

            # ── 3. 语义向量匹配（优先） ──
            if self.config.semantic.enabled and not self._cache.is_empty:
                try:
                    query_text = emotion_hint.strip() if emotion_hint else ""
                    if not query_text and extra_context:
                        # 无 emotion_hint 时用对话上下文作为查询
                        query_text = extra_context

                    if query_text:
                        matched_tag, matched_desc = await self._semantic_select(
                            query_text
                        )
                        if matched_tag and matched_desc:
                            # matched_desc 必须精确命中本轮实际记录；禁止再次使用
                            # get_by_description 模糊查询。
                            emoji_result = desc_to_emoji.get(matched_desc)
                            emoji_base64 = emoji_result.base64_data if emoji_result else ""
                            if emoji_base64:
                                try:
                                    send_result = await self.ctx.send.emoji(
                                        emoji_base64, stream_id
                                    )
                                except Exception as exc:
                                    logger.error(
                                        "[EmojiTextSelector] 语义匹配发送异常: %s",
                                        type(exc).__name__,
                                    )
                                    return self._finish_tool_result(
                                        diagnostics,
                                        success=False,
                                        stage="send",
                                        selection_method="semantic",
                                        error="发送表情包失败",
                                    )
                                if send_result:
                                    return self._finish_tool_result(
                                        diagnostics,
                                        success=True,
                                        stage="complete",
                                        selection_method="semantic",
                                        content=f"表情包发送成功（{matched_desc}）",
                                        description=matched_desc,
                                    )
                                logger.error(
                                    "[EmojiTextSelector] 语义匹配发送失败，不再尝试第二次发送"
                                )
                                return self._finish_tool_result(
                                    diagnostics,
                                    success=False,
                                    stage="send",
                                    selection_method="semantic",
                                    error="发送表情包失败",
                                )
                            logger.error(
                                "[EmojiTextSelector] 语义匹配命中但本轮实际记录不存在，"
                                "将降级到文本 LLM 选择"
                            )
                            diagnostics.add_warning("semantic_stale_match")
                        else:
                            diagnostics.add_warning("semantic_fallback")
                    else:
                        diagnostics.add_warning("semantic_query_empty")
                except Exception as exc:
                    logger.warning(
                        "[EmojiTextSelector] 语义匹配失败，降级为文本 LLM 选择: %s",
                        type(exc).__name__,
                    )
                    diagnostics.add_warning("semantic_error")
            elif self.config.semantic.enabled:
                diagnostics.add_warning("semantic_cache_unavailable")

            # ── 4. 文本 LLM 选择（降级） ──
            diagnostics.method = "text_llm"
            prompt = prompting.build_selection_prompt(
                ordered_descriptions,
                conversation_context=extra_context,
                emotion_expression=emotion_hint,
            )
            try:
                llm_result = await self.ctx.llm.generate(
                    prompt=prompt,
                    model=self.config.selector.llm_model,
                )
            except Exception as exc:
                logger.warning(
                    "[EmojiTextSelector] LLM 选择调用失败: %s",
                    type(exc).__name__,
                )
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="text_select",
                    selection_method="text_llm",
                    error="LLM 选择失败",
                )

            response_text = ""
            if isinstance(llm_result, dict):
                if llm_result.get("success") is False:
                    logger.warning("[EmojiTextSelector] Host 返回 LLM 选择失败")
                    return self._finish_tool_result(
                        diagnostics,
                        success=False,
                        stage="text_select",
                        selection_method="text_llm",
                        error="LLM 选择失败",
                    )
                response_text = str(
                    llm_result.get("response") or llm_result.get("content") or ""
                ).strip()

            selected_idx = prompting.parse_llm_index(response_text, len(ordered_descriptions))

            # 5. 解析 LLM 选择结果
            if selected_idx is None:
                logger.warning(
                    "[EmojiTextSelector] LLM 索引解析失败，放弃发送"
                )
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="text_select",
                    selection_method="text_llm",
                    error="LLM 索引解析失败",
                )

            selected_desc = ordered_descriptions[selected_idx - 1]

            # 6. 获取选中表情包的 base64 并发送
            chosen = desc_to_emoji.get(selected_desc)
            if chosen is None:
                logger.error(
                    f"[EmojiTextSelector] 选中描述[{selected_idx}]无对应实际记录，放弃发送"
                )
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="text_select",
                    selection_method="text_llm",
                    error="选中描述无对应实际记录",
                )
            emoji_base64 = chosen.base64_data
            chosen_desc = chosen.description

            if not emoji_base64:
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="text_select",
                    selection_method="text_llm",
                    error="选中表情包的 base64 数据为空",
                )

            # 7. 发送
            try:
                send_result = await self.ctx.send.emoji(emoji_base64, stream_id)
            except Exception as exc:
                logger.error(
                    "[EmojiTextSelector] 发送表情包异常: %s",
                    type(exc).__name__,
                )
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="send",
                    selection_method="text_llm",
                    error="发送表情包失败",
                )
            if not send_result:
                logger.error("[EmojiTextSelector] 发送表情包失败")
                return self._finish_tool_result(
                    diagnostics,
                    success=False,
                    stage="send",
                    selection_method="text_llm",
                    error="发送表情包失败",
                )

            return self._finish_tool_result(
                diagnostics,
                success=True,
                stage="complete",
                selection_method="text_llm",
                content=f"表情包发送成功（{chosen_desc}）",
                description=chosen_desc,
                selected_index=selected_idx,
            )

        except Exception as exc:
            logger.error(
                "[EmojiTextSelector] 工具执行异常: %s",
                type(exc).__name__,
            )
            return self._finish_tool_result(
                diagnostics,
                success=False,
                stage="execute",
                error="插件执行失败",
            )

    # ─── Hook: 从 planner 工具列表里移除 send_emoji ──────────────

    @HookHandler(
        "maisaka.planner.before_request",
        mode=HookMode.BLOCKING,
    )
    async def filter_send_emoji_tool(self, **kwargs: Any) -> dict[str, Any]:
        """根据配置决定是否从 planner 工具列表中移除内置 send_emoji。"""
        tools = kwargs.get("tool_definitions")
        if not isinstance(tools, list):
            return {"modified_kwargs": kwargs}

        try:
            filter_send = self.config.general.filter_send_emoji
        except RuntimeError:
            logger.warning(
                "[EmojiTextSelector] 配置未注入，跳过工具过滤"
            )
            return {"modified_kwargs": kwargs}

        if not filter_send:
            return {"modified_kwargs": kwargs}

        before_count = len(tools)
        filtered_tools = [
            t for t in tools
            if not (
                isinstance(t, dict)
                and t.get("function", {}).get("name") == "send_emoji"
            )
        ]
        after_count = len(filtered_tools)
        logger.info(
            f"[EmojiTextSelector] 工具过滤: {before_count} → {after_count}, "
            f"移除: send_emoji, filter_send_emoji={filter_send}"
        )
        return {"modified_kwargs": {**kwargs, "tool_definitions": filtered_tools}}


def create_plugin() -> EmojiTextSelectorPlugin:
    """插件工厂函数，由 SDK Runner 调用以创建插件实例。"""
    return EmojiTextSelectorPlugin()
