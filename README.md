# Maibot_Emoji_Select_With_Text

[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](./LICENSE)

一个根据文字描述选择表情包的 MaiBot 插件。不需要为选表情单独配置视觉模型，只要表情包已经有描述，就可以结合当前对话挑选并发送。

## 功能

- 注册 `select_emoji_with_text` 工具，供 Planner 在聊天中直接调用。
- 结合 `emotion_hint` 和最近的聊天内容判断当前语气。
- 选择和发送使用同一份表情包快照，避免表情顺序变化后发错图片。
- 每次最多发送一张。选择、图片读取或发送失败时会停止，不会随机补发。
- 可以隐藏 MaiBot 原生的 `send_emoji`，让表情发送统一经过本插件。
- 可选使用 embedding 做语义匹配；不可用或相似度不足时自动回到文本选择。

## 兼容性

插件清单声明支持：

- MaiBot `1.0.0`–`1.9.9`
- maibot-plugin-sdk `2.7.1`
- Python `3.12+`

当前维护基线是 MaiBot `1.1.0` 与 SDK `2.7.1`。MaiBot `1.0.0` 也在声明范围内，但官方 `1.0.0` 环境如果仍使用较旧的 SDK，需要先升级到 `2.7.1` 才能加载本插件。

`1.2.0`–`1.9.9` 是预留的 MaiBot 1.x 兼容范围，会随正式版本发布继续复查。MaiBot 2.x 不在当前支持范围内。

## 安装

进入 MaiBot 的插件目录并克隆仓库：

```bash
cd /path/to/MaiBot/plugins
git clone https://github.com/TogawaSakiko-desuwa/Maibot_Emoji_Select_With_Text.git
```

然后重启 MaiBot，或者在支持热重载的 WebUI 中重新加载插件。

如果使用 WebUI 的仓库安装功能，请填写：

- 仓库：`https://github.com/TogawaSakiko-desuwa/Maibot_Emoji_Select_With_Text`
- 分支：`main`
- 插件 ID：`togawasakiko-desuwa.emoji-select-with-text`

使用前请先在 MaiBot 中导入表情包并生成描述。没有描述或没有图片数据的记录不会参与选择。

## 使用

插件不增加聊天命令。启用后，Planner 会在需要发送表情时调用 `select_emoji_with_text`。

可以用一句意图明确的话确认效果，例如：

```text
用一个表情表达“听完这句话有点无奈”。
```

默认配置采用文本模型选择，安装后即可使用。文本选择的流程是：读取表情描述、建立编号列表、让模型返回一个编号，再从同一份快照中发送对应图片。

## 配置

```toml
[general]
filter_send_emoji = true

[selector]
max_emotion_tags = 0
llm_model = "utils"
context_message_limit = 30

[semantic]
enabled = false
refresh_interval_seconds = 300
similarity_threshold = 0.3
embed_batch_size = 64
embedding_task = "embedding"
```

| 配置项 | 默认值 | 说明 |
| --- | ---: | --- |
| `general.filter_send_emoji` | `true` | 从 Planner 工具列表中隐藏原生 `send_emoji` |
| `selector.max_emotion_tags` | `0` | 最多提供多少条表情描述；`0` 表示不限制 |
| `selector.llm_model` | `"utils"` | 文本选择使用的 MaiBot 模型任务名 |
| `selector.context_message_limit` | `30` | 最多读取多少条最近消息 |
| `semantic.enabled` | `false` | 是否优先使用 embedding 语义匹配 |
| `semantic.refresh_interval_seconds` | `300` | 语义缓存刷新间隔，单位为秒 |
| `semantic.similarity_threshold` | `0.3` | 允许语义命中的最低余弦相似度 |
| `semantic.embed_batch_size` | `64` | 每批生成向量的描述数量 |
| `semantic.embedding_task` | `"embedding"` | 生成向量使用的 MaiBot 模型任务名 |

`llm_model` 和 `embedding_task` 填写的都是 MaiBot 配置中的任务名，不是模型厂商提供的模型 ID。

## 语义匹配

将 `semantic.enabled` 改为 `true` 后，插件会为可用的表情描述建立向量缓存。缓存保存在 MaiBot 分配给插件的数据目录中，文件名为 `emoji_semantic_cache.npz`，不会写入插件安装目录。

缓存记录了描述、embedding 任务、实际模型名称和向量维度。任务、模型或维度发生变化时会完整重建；三者不变时，会按描述复用已有向量。新快照完整生成并写入成功后才会替换旧快照，因此刷新失败不会破坏上一次可用结果。MaiBot 明确返回空表情库时，缓存会自动清理。

实际选择时，语义匹配出现异常、缓存尚未就绪或结果低于阈值，都会自动降级到文本模型，不需要手动删除缓存。

## 常见问题

### Planner 中看不到工具

确认插件已经启用，并检查 MaiBot、SDK 和 Python 版本。正常情况下工具名为 `select_emoji_with_text`；修改配置后，请等待 MaiBot 完成插件重载。

### 表情包库中没有可用记录

确认 MaiBot 已导入表情包，并且记录同时具有非空描述和图片数据。重复描述只保留第一条有效记录。

### LLM 选择失败或索引解析失败

检查 `selector.llm_model` 对应的模型任务是否存在，并能正常完成文本生成。默认任务名是 `utils`。

### 语义模式一直回到文本选择

先确认 `semantic.embedding_task` 对应的任务可以生成 embedding。插件日志中的诊断字段会区分缓存未就绪、模型身份变化、低于阈值和调用异常；这些情况都不会影响文本选择继续工作。

### 发送表情包失败

插件不会在发送失败后换一张再次尝试。请确认消息流仍然有效，并查看 MaiBot 日志中 `send.emoji` 的错误信息。

## 许可证

本项目使用 GPL-3.0-or-later，完整条款见 [LICENSE](./LICENSE)。来源归属与修改说明见 [NOTICE](./NOTICE)。
