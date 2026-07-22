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
from typing import Any, ClassVar

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
        self._refresh_task: asyncio.Task | None = None
        self._refresh_lock = asyncio.Lock()
        self._refresh_wakeup = asyncio.Event()
        self._refresh_requested = False
        self._plugin_dir: Path | None = None
        self._semantic_task_name: str | None = None
        self._semantic_candidate_limit: int | None = None

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
        self._semantic_task_name = self.config.semantic.embedding_task
        self._semantic_candidate_limit = self.config.selector.max_emotion_tags

        if self.config.semantic.enabled:
            restored = emoji_cache.EmojiEmbeddingCache.load(self._cache_path())
            if restored is not None:
                self._cache = restored
                logger.info(
                    "[EmojiTextSelector] 从磁盘恢复向量缓存成功，共 %d 条",
                    self._cache.count,
                )
            if (
                not self._cache.is_empty
                and self._cache.identity is not None
                and self._cache.identity.task_name != self.config.semantic.embedding_task
            ):
                self._request_refresh()
            self._start_refresh_task()
        logger.info("[EmojiTextSelector] 插件已加载")

    async def on_unload(self) -> None:
        await self._stop_refresh_task()
        logger.info("[EmojiTextSelector] 插件已卸载")

    async def on_config_update(
        self, scope: str, config_data: dict[str, Any], version: str
    ) -> None:
        del config_data, version
        if scope != "self":
            return
        worker_was_running = self._refresh_task is not None and not self._refresh_task.done()
        current_task_name = self.config.semantic.embedding_task
        current_candidate_limit = self.config.selector.max_emotion_tags
        task_changed = (
            self._semantic_task_name is not None
            and self._semantic_task_name != current_task_name
        )
        candidate_limit_changed = (
            self._semantic_candidate_limit is not None
            and self._semantic_candidate_limit != current_candidate_limit
        )
        self._semantic_task_name = current_task_name
        self._semantic_candidate_limit = current_candidate_limit
        if self.config.semantic.enabled:
            if task_changed or candidate_limit_changed:
                await self._stop_refresh_task()
                self._request_refresh()
            elif not worker_was_running:
                self._request_refresh()
            if self._cache.is_empty:
                restored = emoji_cache.EmojiEmbeddingCache.load(self._cache_path())
                if restored is not None:
                    self._cache = restored
            self._start_refresh_task()
            self._refresh_wakeup.set()
        else:
            await self._stop_refresh_task()

    def _start_refresh_task(self) -> None:
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._background_refresh_loop())
        self._refresh_wakeup.set()

    def _request_refresh(self) -> None:
        self._refresh_requested = True
        self._refresh_wakeup.set()

    async def _stop_refresh_task(self) -> None:
        task = self._refresh_task
        self._refresh_task = None
        if task is None:
            return
        self._refresh_wakeup.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _cache_path(self) -> Path:
        return Path(self.ctx.paths.data_dir) / emoji_cache.CACHE_FILE_NAME

    def _cleanup_legacy_cache(self) -> None:
        if self._plugin_dir is None:
            return
        legacy_directory = self._plugin_dir / ".cache"
        for filename in emoji_cache.LEGACY_CACHE_FILE_NAMES:
            try:
                (legacy_directory / filename).unlink(missing_ok=True)
            except OSError as error:
                logger.warning(
                    "[EmojiTextSelector] 清理旧缓存失败: %s",
                    type(error).__name__,
                )

    async def _read_emoji_snapshot(self) -> list[object] | None:
        """Read and normalize one Host emoji-library response."""

        try:
            raw_records = await self.ctx.emoji.get_all()
        except Exception as exc:
            logger.warning(
                "[EmojiTextSelector] 读取表情包库异常: %s",
                type(exc).__name__,
            )
            return None

        if isinstance(raw_records, list):
            return raw_records
        if (
            isinstance(raw_records, dict)
            and raw_records.get("success") is True
            and isinstance(raw_records.get("emojis"), list)
        ):
            return raw_records["emojis"]

        logger.warning(
            "[EmojiTextSelector] Host 返回了不可用的表情包库结果: %s",
            type(raw_records).__name__,
        )
        return None

    async def _send_emoji_once(
        self,
        base64_data: str,
        stream_id: str,
        *,
        method: str,
    ) -> bool:
        """Send exactly once and normalize SDK or defensive envelope results."""

        try:
            result = await self.ctx.send.emoji(base64_data, stream_id)
        except Exception as exc:
            logger.error(
                "[EmojiTextSelector] 发送表情包异常: method=%s error=%s",
                method,
                type(exc).__name__,
            )
            return False

        success = (
            result.get("success") is True
            if isinstance(result, dict)
            else bool(result)
        )
        if not success:
            logger.error(
                "[EmojiTextSelector] 发送表情包失败: method=%s",
                method,
            )
        return success

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
        while True:
            try:
                self._refresh_wakeup.clear()
                retry_after_timeout = False
                if not self.config.semantic.enabled:
                    return
                interval = self.config.semantic.refresh_interval_seconds
                identity = self._cache.identity
                task_mismatch = (
                    not self._cache.is_empty
                    and (identity is None or identity.task_name != self.config.semantic.embedding_task)
                )
                if self._refresh_requested or task_mismatch or self._cache.needs_refresh(interval):
                    self._refresh_requested = False
                    refreshed = await self._refresh_cache()
                    if refreshed:
                        delay = interval
                    else:
                        delay = 30
                        retry_after_timeout = True
                else:
                    elapsed = max(0.0, time.time() - self._cache.refreshed_at)
                    delay = max(0.1, interval - elapsed)

                if self._refresh_requested:
                    continue
                try:
                    await asyncio.wait_for(self._refresh_wakeup.wait(), timeout=delay)
                except TimeoutError:
                    if retry_after_timeout:
                        self._refresh_requested = True
                else:
                    if retry_after_timeout:
                        self._refresh_requested = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "[EmojiTextSelector] 向量缓存刷新循环异常: %s",
                    type(exc).__name__,
                )
                await asyncio.sleep(30)

    async def _refresh_cache(self) -> bool:
        """Build a complete staging snapshot and commit it transactionally."""

        async with self._refresh_lock:
            refresh_start = time.perf_counter()
            task_name = self.config.semantic.embedding_task
            candidate_limit = self.config.selector.max_emotion_tags
            if not self.config.semantic.enabled:
                return False
            raw_records = await self._read_emoji_snapshot()
            if raw_records is None:
                return False

            if not raw_records:
                if not self._semantic_config_matches(task_name, candidate_limit):
                    return False
                try:
                    self._cache_path().unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "[EmojiTextSelector] 清理空表情库缓存失败: %s",
                        type(exc).__name__,
                    )
                    return False
                self._cache = emoji_cache.EmojiEmbeddingCache.empty(
                    refreshed_at=time.time()
                )
                self._cleanup_legacy_cache()
                logger.info("[EmojiTextSelector] 表情包库为空，已清理语义缓存")
                return True

            candidate_set = selection.prepare_candidates(
                raw_records,
                limit=candidate_limit,
            )
            descriptions = [candidate.description for candidate in candidate_set.candidates]
            if not descriptions:
                logger.warning("[EmojiTextSelector] 表情记录中没有可构建缓存的有效候选")
                return False

            try:
                probe_result = await self.ctx.llm.embed(
                    text=descriptions[0],
                    task_name=task_name,
                )
                probe_vector, model_name = emoji_cache.parse_embedding_vector(
                    probe_result
                )
                identity = emoji_cache.EmbeddingIdentity(
                    task_name=task_name,
                    model_name=model_name,
                    dimension=int(probe_vector.size),
                )

                reusable = self._cache.reusable_vectors(descriptions, identity)
                vectors_by_description = dict(reusable)
                vectors_by_description[descriptions[0]] = probe_vector
                pending_descriptions = [
                    description
                    for description in descriptions[1:]
                    if description not in reusable
                ]

                batch_size = self.config.semantic.embed_batch_size
                for batch_start in range(0, len(pending_descriptions), batch_size):
                    batch = pending_descriptions[batch_start:batch_start + batch_size]
                    batch_result = await self.ctx.llm.embed(
                        texts=batch,
                        task_name=task_name,
                    )
                    if not isinstance(batch_result, dict) or batch_result.get("success") is not True:
                        raise ValueError("批量 embedding 调用失败")
                    results = batch_result.get("results")
                    if not isinstance(results, list) or len(results) != len(batch):
                        raise ValueError("批量 embedding 结果数量不一致")
                    for description, result in zip(batch, results, strict=True):
                        vector, _ = emoji_cache.parse_embedding_vector(
                            result,
                            expected_model_name=identity.model_name,
                            expected_dimension=identity.dimension,
                        )
                        vectors_by_description[description] = vector

                matrix = np.vstack(
                    [vectors_by_description[description] for description in descriptions]
                )
                staging = emoji_cache.EmojiEmbeddingCache.build(
                    descriptions=descriptions,
                    vectors=matrix,
                    identity=identity,
                    refreshed_at=time.time(),
                )
                if not self._semantic_config_matches(task_name, candidate_limit):
                    return False
                staging.save_atomic(self._cache_path())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[EmojiTextSelector] 语义缓存刷新未提交: %s",
                    type(exc).__name__,
                )
                return False

            self._cache = staging
            self._cleanup_legacy_cache()
            elapsed_ms = round((time.perf_counter() - refresh_start) * 1_000)
            logger.info(
                "[EmojiTextSelector] 向量缓存刷新完成: count=%d reused=%d "
                "duration_ms=%d",
                self._cache.count,
                len(reusable),
                elapsed_ms,
            )
            return True

    def _semantic_config_matches(self, task_name: str, candidate_limit: int) -> bool:
        return (
            self.config.semantic.enabled
            and self.config.semantic.embedding_task == task_name
            and self.config.selector.max_emotion_tags == candidate_limit
        )

    async def _semantic_select(
        self,
        query_text: str,
    ) -> tuple[str | None, str | None]:
        """Return ``(description, warning_code)`` for one semantic query."""

        if not query_text.strip():
            return None, "semantic_query_empty"

        task_name = self.config.semantic.embedding_task
        cache_snapshot = self._cache
        identity = cache_snapshot.identity
        if identity is None or identity.task_name != task_name:
            self._request_refresh()
            return None, "semantic_cache_mismatch"

        try:
            embed_result = await self.ctx.llm.embed(
                text=query_text,
                task_name=task_name,
            )
        except Exception as exc:
            logger.warning(
                "[EmojiTextSelector] 查询 embedding 调用失败: %s",
                type(exc).__name__,
            )
            return None, "semantic_error"

        if not isinstance(embed_result, dict) or embed_result.get("success") is not True:
            logger.warning("[EmojiTextSelector] 查询 embedding 失败")
            return None, "semantic_api_error"

        try:
            query_vector, model_name = emoji_cache.parse_embedding_vector(embed_result)
        except ValueError:
            logger.warning("[EmojiTextSelector] 查询 embedding 返回了无效向量")
            return None, "semantic_invalid_vector"

        query_identity = emoji_cache.EmbeddingIdentity(
            task_name=task_name,
            model_name=model_name,
            dimension=int(query_vector.size),
        )
        if query_identity != identity:
            self._request_refresh()
            logger.info("[EmojiTextSelector] 查询 embedding 身份已变化，等待缓存重建")
            return None, "semantic_cache_mismatch"

        query_vector = query_vector / np.linalg.norm(query_vector)
        try:
            top_matches = cache_snapshot.search_descriptions(
                query_vector,
                self.config.semantic.similarity_threshold,
                max_count=1,
            )
        except ValueError:
            self._request_refresh()
            return None, "semantic_cache_mismatch"

        if not top_matches:
            logger.info("[EmojiTextSelector] 语义匹配未找到超过阈值的表情包")
            return None, "semantic_below_threshold"

        matched_description, best_score = top_matches[0]
        logger.info(
            "[EmojiTextSelector] 语义匹配命中: score=%.3f",
            best_score,
        )
        return matched_description, None

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
            raw_emoji_records = await self._read_emoji_snapshot()
            if raw_emoji_records is None:
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
                        matched_desc, semantic_warning = await self._semantic_select(
                            query_text
                        )
                        if semantic_warning:
                            diagnostics.add_warning(semantic_warning)
                        if matched_desc:
                            # matched_desc 必须精确命中本轮实际记录；禁止再次使用
                            # get_by_description 模糊查询。
                            emoji_result = desc_to_emoji.get(matched_desc)
                            emoji_base64 = emoji_result.base64_data if emoji_result else ""
                            if emoji_base64:
                                if await self._send_emoji_once(
                                    emoji_base64,
                                    stream_id,
                                    method="semantic",
                                ):
                                    return self._finish_tool_result(
                                        diagnostics,
                                        success=True,
                                        stage="complete",
                                        selection_method="semantic",
                                        content=f"表情包发送成功（{matched_desc}）",
                                        description=matched_desc,
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
            if not await self._send_emoji_once(
                emoji_base64,
                stream_id,
                method="text_llm",
            ):
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
