"""Validated in-memory and on-disk cache for emoji description embeddings."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


logger = logging.getLogger("Maibot_Emoji_Select_With_Text")
CACHE_FILE_NAME = "emoji_semantic_cache.npz"
CACHE_SCHEMA_VERSION = 2
LEGACY_CACHE_FILE_NAMES = (
    "emoji_vector_cache.npz",
    "emoji_vector_cache_meta.json",
)


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """The Host embedding identity required to reuse cached vectors."""

    task_name: str
    model_name: str
    dimension: int

    def __post_init__(self) -> None:
        task_name = self.task_name.strip()
        model_name = self.model_name.strip()
        if not task_name or not model_name:
            raise ValueError("embedding 任务名和模型名不能为空")
        if self.dimension < 1:
            raise ValueError("embedding 向量维度必须大于 0")
        object.__setattr__(self, "task_name", task_name)
        object.__setattr__(self, "model_name", model_name)


def parse_embedding_vector(
    payload: object,
    *,
    expected_model_name: str | None = None,
    expected_dimension: int | None = None,
) -> tuple[np.ndarray, str]:
    """Validate one Host embedding payload without retaining the raw response."""

    if not isinstance(payload, dict) or payload.get("success") is False:
        raise ValueError("Host 未返回可用的 embedding")
    model_name = str(payload.get("model_name") or payload.get("model") or "").strip()
    if not model_name:
        raise ValueError("Host 未返回 embedding 模型名")
    try:
        vector = np.asarray(payload.get("embedding"), dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Host 返回的 embedding 无法解析") from error
    if vector.ndim != 1 or vector.size < 1:
        raise ValueError("Host 返回的 embedding 不是一维向量")
    with np.errstate(over="ignore", invalid="ignore"):
        norm = np.linalg.norm(vector)
    if (
        not np.isfinite(vector).all()
        or not np.isfinite(norm)
        or norm < 1e-12
    ):
        raise ValueError("Host 返回的 embedding 含无效数值")
    if expected_model_name is not None and model_name != expected_model_name:
        raise ValueError("批次 embedding 模型不一致")
    if expected_dimension is not None and vector.size != expected_dimension:
        raise ValueError("批次 embedding 维度不一致")
    return vector, model_name


class EmojiEmbeddingCache:
    """One complete, validated embedding snapshot."""

    def __init__(self) -> None:
        self._descriptions: tuple[str, ...] = ()
        self._matrix: np.ndarray | None = None
        self._identity: EmbeddingIdentity | None = None
        self._refreshed_at = 0.0

    @classmethod
    def build(
        cls,
        *,
        descriptions: list[str],
        vectors: np.ndarray,
        identity: EmbeddingIdentity,
        refreshed_at: float,
    ) -> EmojiEmbeddingCache:
        """Build and validate a complete cache snapshot."""

        cleaned = [description.strip() for description in descriptions]
        if not cleaned or any(not description for description in cleaned):
            raise ValueError("缓存描述不能为空")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("缓存描述不能重复")

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (len(cleaned), identity.dimension):
            raise ValueError("embedding 矩阵形状与缓存身份不一致")
        if not np.isfinite(matrix).all():
            raise ValueError("embedding 矩阵包含非有限数值")
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms < 1e-12):
            raise ValueError("embedding 矩阵包含无效或零向量")
        if not np.isfinite(refreshed_at) or refreshed_at < 0:
            raise ValueError("缓存刷新时间无效")

        cache = cls()
        cache._descriptions = tuple(cleaned)
        cache._matrix = (matrix / norms).astype(np.float32)
        cache._identity = identity
        cache._refreshed_at = float(refreshed_at)
        return cache

    @classmethod
    def empty(cls, *, refreshed_at: float = 0.0) -> EmojiEmbeddingCache:
        """Create an empty snapshot, optionally marking an explicit empty refresh."""

        if not np.isfinite(refreshed_at) or refreshed_at < 0:
            raise ValueError("缓存刷新时间无效")
        cache = cls()
        cache._refreshed_at = float(refreshed_at)
        return cache

    @property
    def is_empty(self) -> bool:
        return not self._descriptions

    @property
    def count(self) -> int:
        return len(self._descriptions)

    @property
    def descriptions(self) -> tuple[str, ...]:
        return self._descriptions

    @property
    def identity(self) -> EmbeddingIdentity | None:
        return self._identity

    @property
    def refreshed_at(self) -> float:
        return self._refreshed_at

    def needs_refresh(self, interval_seconds: int) -> bool:
        return (time.time() - self._refreshed_at) > interval_seconds

    def reusable_vectors(
        self,
        descriptions: list[str],
        identity: EmbeddingIdentity,
    ) -> dict[str, np.ndarray]:
        """Return copies of reusable rows only when the full identity matches."""

        if self._identity != identity or self._matrix is None:
            return {}
        by_description = {
            description: self._matrix[index].copy()
            for index, description in enumerate(self._descriptions)
        }
        return {
            description: by_description[description]
            for description in descriptions
            if description in by_description
        }

    def search_descriptions(
        self,
        query_vector: np.ndarray,
        threshold: float,
        max_count: int,
    ) -> list[tuple[str, float]]:
        """Return matching descriptions in descending cosine-similarity order."""

        if self._matrix is None or max_count < 1:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._matrix.shape[1]:
            raise ValueError("查询向量维度与缓存不一致")
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("查询向量无效")
        scores = self._matrix @ (query / norm)
        eligible_indices = np.flatnonzero(scores >= threshold)
        if eligible_indices.size == 0:
            return []
        ordered_indices = eligible_indices[np.argsort(-scores[eligible_indices])]
        return [
            (self._descriptions[int(index)], float(scores[int(index)]))
            for index in ordered_indices[:max_count]
        ]

    def save_atomic(self, cache_path: Path) -> None:
        """Persist one schema-v2 snapshot with a same-directory atomic replace."""

        if self._matrix is None or self._identity is None or self.is_empty:
            raise ValueError("空缓存不能持久化")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            dir=cache_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                np.savez_compressed(
                    temporary_file,
                    schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
                    refreshed_at=np.asarray(self._refreshed_at, dtype=np.float64),
                    embedding_task=np.asarray(self._identity.task_name),
                    model_name=np.asarray(self._identity.model_name),
                    dimension=np.asarray(self._identity.dimension, dtype=np.int64),
                    descriptions=np.asarray(self._descriptions, dtype=np.str_),
                    vectors=self._matrix,
                )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, cache_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, cache_path: Path) -> EmojiEmbeddingCache | None:
        """Load and strictly validate one schema-v2 snapshot."""

        if not cache_path.exists():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                schema_version = int(np.asarray(payload["schema_version"]).item())
                refreshed_at = float(np.asarray(payload["refreshed_at"]).item())
                task_name = str(np.asarray(payload["embedding_task"]).item())
                model_name = str(np.asarray(payload["model_name"]).item())
                dimension = int(np.asarray(payload["dimension"]).item())
                raw_descriptions = payload["descriptions"]
                if raw_descriptions.ndim != 1:
                    raise ValueError("缓存描述数组维度无效")
                descriptions = [str(item) for item in raw_descriptions.tolist()]
                vectors = payload["vectors"].astype(np.float32)
            if schema_version != CACHE_SCHEMA_VERSION:
                return None
            return cls.build(
                descriptions=descriptions,
                vectors=vectors,
                identity=EmbeddingIdentity(task_name, model_name, dimension),
                refreshed_at=refreshed_at,
            )
        except (OSError, EOFError, BadZipFile, ValueError, TypeError, KeyError) as error:
            logger.warning("加载语义缓存失败: %s", type(error).__name__)
            return None
