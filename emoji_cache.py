"""In-memory and on-disk embedding cache for emoji descriptions."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import logging
import time

import numpy as np


logger = logging.getLogger("Maibot_Emoji_Select_With_Text")
_VECTOR_CACHE_FILE = "emoji_vector_cache.npz"
_VECTOR_CACHE_META_FILE = "emoji_vector_cache_meta.json"


class EmojiEmbeddingCache:
    """表情包描述 embedding 向量缓存，使用 numpy 矩阵加速检索。"""

    def __init__(self) -> None:
        self._ids: np.ndarray = np.array([], dtype=np.int64)
        self._text_keys: Dict[int, str] = {}
        self._emotion_tags: Dict[int, str] = {}
        self._tag_to_desc: Dict[str, str] = {}
        self._matrix: Optional[np.ndarray] = None
        self._last_refresh_time = 0.0
        self._refreshing = False

    @property
    def is_empty(self) -> bool:
        return len(self._ids) == 0

    @property
    def count(self) -> int:
        return len(self._ids)

    def needs_refresh(self, interval_seconds: int) -> bool:
        return not self._refreshing and (time.time() - self._last_refresh_time) > interval_seconds

    def rebuild(
        self,
        ids: List[int],
        text_keys: Dict[int, str],
        emotion_tags: Dict[int, str],
        matrix: np.ndarray,
    ) -> None:
        if not ids:
            self._ids = np.array([], dtype=np.int64)
            self._text_keys = {}
            self._emotion_tags = {}
            self._tag_to_desc = {}
            self._matrix = None
            return
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("embedding 矩阵行数必须与 id 数量一致")
        self._ids = np.array(ids, dtype=np.int64)
        self._text_keys = dict(text_keys)
        self._emotion_tags = dict(emotion_tags)
        self._rebuild_tag_descriptions()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self._matrix = (matrix / norms).astype(np.float32)

    def _rebuild_tag_descriptions(self) -> None:
        tag_to_description: Dict[str, str] = {}
        seen_descriptions: set[str] = set()
        for cache_id in self._ids:
            identifier = int(cache_id)
            tag = self._emotion_tags.get(identifier)
            description = self._text_keys.get(identifier)
            if tag and description and description not in seen_descriptions:
                seen_descriptions.add(description)
                tag_to_description[tag] = description
        self._tag_to_desc = tag_to_description

    def get_text_key(self, expr_id: int) -> Optional[str]:
        return self._text_keys.get(expr_id)

    def get_emotion_tag(self, expr_id: int) -> Optional[str]:
        return self._emotion_tags.get(expr_id)

    def get_tag_to_id_map(self) -> Dict[str, int]:
        return {value: int(key) for key, value in self._emotion_tags.items()}

    def get_tag_description_map(self) -> Dict[str, str]:
        return dict(self._tag_to_desc)

    def get_existing_entries(
        self,
        valid_ids: set[int],
        changed_ids: set[int],
    ) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[int, str]]:
        if self._matrix is None or len(self._ids) == 0:
            return [], np.empty((0, 0), dtype=np.float32), {}, {}
        mask = np.array(
            [int(identifier) in valid_ids and int(identifier) not in changed_ids for identifier in self._ids],
            dtype=bool,
        )
        kept_ids = [int(identifier) for identifier in self._ids[mask].tolist()]
        return (
            kept_ids,
            self._matrix[mask],
            {identifier: self._text_keys[identifier] for identifier in kept_ids if identifier in self._text_keys},
            {identifier: self._emotion_tags[identifier] for identifier in kept_ids if identifier in self._emotion_tags},
        )

    def mark_refreshed(self) -> None:
        self._last_refresh_time = time.time()
        self._refreshing = False

    def set_refreshing(self) -> None:
        self._refreshing = True

    def search(self, query_vector: np.ndarray, threshold: float, max_count: int) -> List[Tuple[int, float]]:
        if self._matrix is None or len(self._ids) == 0 or max_count < 1:
            return []
        if query_vector.ndim != 1 or self._matrix.shape[1] != query_vector.shape[0]:
            raise ValueError("查询向量维度与缓存不一致")
        scores = self._matrix @ query_vector
        eligible = scores >= threshold
        if not eligible.any():
            return []
        filtered_scores = scores[eligible]
        filtered_ids = self._ids[eligible]
        top_indices = np.argsort(-filtered_scores)[:max_count]
        return [(int(filtered_ids[index]), float(filtered_scores[index])) for index in top_indices]

    def save_to_disk(self, cache_dir: Path) -> None:
        if self._matrix is None or len(self._ids) == 0:
            return
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_dir / _VECTOR_CACHE_FILE, ids=self._ids, matrix=self._matrix)
            metadata = {
                "schema_version": 1,
                "text_keys": {str(key): value for key, value in self._text_keys.items()},
                "emotion_tags": {str(key): value for key, value in self._emotion_tags.items()},
            }
            (cache_dir / _VECTOR_CACHE_META_FILE).write_text(
                json.dumps(metadata, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, ValueError, TypeError) as error:
            logger.warning("向量缓存持久化失败: %s", error)

    def load_from_disk(self, cache_dir: Path) -> bool:
        npz_path = cache_dir / _VECTOR_CACHE_FILE
        metadata_path = cache_dir / _VECTOR_CACHE_META_FILE
        if not npz_path.exists() or not metadata_path.exists():
            return False
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                ids = data["ids"].astype(np.int64)
                matrix = data["matrix"].astype(np.float32)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version", 1) != 1:
                return False
            text_keys = {int(key): str(value) for key, value in metadata.get("text_keys", {}).items()}
            emotion_tags = {int(key): str(value) for key, value in metadata.get("emotion_tags", {}).items()}
            self.rebuild([int(item) for item in ids.tolist()], text_keys, emotion_tags, matrix)
            self._last_refresh_time = time.time()
            return True
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            logger.warning("从磁盘加载向量缓存失败: %s", error)
            return False
