"""Configuration models for the emoji selector plugin."""

from maibot_sdk import PluginConfigBase
from pydantic import Field, field_validator


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"

    enabled: bool = Field(default=True, description="是否启用插件", json_schema_extra={"label": "启用"})
    config_version: str = Field(
        default="1.1.0",
        description="配置版本号",
        json_schema_extra={"label": "配置版本"},
    )


class GeneralSectionConfig(PluginConfigBase):
    """插件通用配置。"""

    __ui_label__ = "通用设置"

    filter_send_emoji: bool = Field(
        default=True,
        description="启用后从 planner 工具列表移除内置 send_emoji，避免 LLM 绕过本插件直接发送",
        json_schema_extra={"label": "过滤原生 send_emoji"},
    )


class EmojiSelectorSectionConfig(PluginConfigBase):
    """LLM 文本选择配置。"""

    __ui_label__ = "LLM 选择"

    max_emotion_tags: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="传给 LLM 的最大情绪标签数量，0 表示不限制",
        json_schema_extra={"label": "最大情绪标签数"},
    )
    llm_model: str = Field(
        default="utils",
        min_length=1,
        description="标签选择用的模型任务名",
        json_schema_extra={"label": "LLM 模型"},
    )
    context_message_limit: int = Field(
        default=30,
        ge=1,
        le=100,
        description="获取最近对话上下文的最大消息数量",
        json_schema_extra={"label": "上下文消息数"},
    )

    @field_validator("llm_model", mode="before")
    @classmethod
    def normalize_llm_model(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("LLM 模型任务名不能为空")
        return value.strip()


class SemanticSectionConfig(PluginConfigBase):
    """语义向量匹配配置。"""

    __ui_label__ = "语义匹配"
    __ui_icon__ = "search"
    __ui_order__ = 3

    enabled: bool = Field(
        default=False,
        description="启用后优先使用 embedding 向量匹配选择表情包，失败时降级为文本 LLM 选择",
        json_schema_extra={"label": "启用语义匹配"},
    )
    refresh_interval_seconds: int = Field(
        default=300,
        ge=30,
        le=86400,
        description="向量缓存刷新间隔（秒）",
        json_schema_extra={"label": "缓存刷新间隔"},
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="最低余弦相似度阈值，低于此值的表情包不会被选中",
        json_schema_extra={"label": "相似度阈值"},
    )
    embed_batch_size: int = Field(
        default=64,
        ge=1,
        le=256,
        description="每批 embedding 请求的最大文本数量",
        json_schema_extra={"label": "Embedding 批次大小"},
    )
    embedding_task: str = Field(
        default="embedding",
        min_length=1,
        description="生成语义向量时使用的模型任务名",
        json_schema_extra={"label": "Embedding 任务"},
    )

    @field_validator("embedding_task", mode="before")
    @classmethod
    def normalize_embedding_task(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Embedding 任务名不能为空")
        return value.strip()


class EmojiTextSelectorConfig(PluginConfigBase):
    """表情包文本选择器配置。"""

    __ui_label__ = "Maibot_Emoji_Select_With_Text"

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    general: GeneralSectionConfig = Field(default_factory=GeneralSectionConfig)
    selector: EmojiSelectorSectionConfig = Field(default_factory=EmojiSelectorSectionConfig)
    semantic: SemanticSectionConfig = Field(default_factory=SemanticSectionConfig)
