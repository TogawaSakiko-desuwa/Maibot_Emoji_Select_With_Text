# Maibot_Emoji_Select_With_Text

[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](./LICENSE)

不给 MaiBot 配视觉模型，也能让它挑对表情包。

如果你的表情库已经有文字描述，这个插件会根据当前对话和想表达的情绪，交给文本模型选出一张合适的表情。默认使用文本选择；embedding 语义匹配也保留了开关，但目前仍是实验功能。

## 能做什么

- 提供独立工具 `select_emoji_with_text`，避免与其他表情选择工具重名。
- 根据 Planner 给出的 `emotion_hint` 选择；没有明确提示时会参考最近聊天记录。
- 一次只发送一张；选不出来就返回错误，不随机抓一张凑数。
- 可以隐藏 MaiBot 原生的 `send_emoji`，避免 Planner 绕过本插件。
- 可选使用 embedding 缓存做语义匹配，匹配失败时回到文本选择。

## 兼容性

插件清单声明的运行环境：

- MaiBot `1.0.0`–`1.9.9`
- maibot-plugin-sdk `2.7.1`
- Python `3.12+`

离线契约检查覆盖了 MaiBot `1.0.0` 和 `1.1.0`，两者都按 SDK `2.7.1` 验证。

`1.2.0`–`1.9.9` 是为后续 1.x 版本预留的兼容上限，新版本发布后仍需要重新验证。MaiBot 2.x 不在当前兼容范围内。

如果 MaiBot 报告 SDK 版本不兼容，请先确认它实际加载的是 `maibot-plugin-sdk 2.7.1`；本插件不声明对 SDK 2.5.x 或 2.6.x 的兼容性。

## 安装

最稳妥的方式是把仓库克隆到 MaiBot 的 `plugins` 目录：

```bash
cd /path/to/MaiBot/plugins
git clone https://github.com/TogawaSakiko-desuwa/Maibot_Emoji_Select_With_Text.git
```

克隆后重启 MaiBot；如果当前 WebUI 支持插件热重载，也可以直接重载。

如果你的 WebUI 提供“从仓库安装”，请填写仓库地址、`main` 分支和插件 ID `togawasakiko-desuwa.emoji-select-with-text`。

表情包需要先在 MaiBot 中正常导入并生成描述；只有图片、没有描述的记录无法参与文本选择。

## 使用

不需要额外的聊天命令。正常聊天时，Planner 会在合适的语境下调用 `select_emoji_with_text`。

想先确认插件有没有工作，可以对麦麦说一句比较明确的话，例如：

```text
用一个表情表达“听完这句话有点无奈”。
```

如果 `filter_send_emoji = true`，Planner 不会再直接调用 MaiBot 原生的 `send_emoji`。

## 配置

默认配置可以直接使用：

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
```

常用配置：

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `filter_send_emoji` | `true` | 从 Planner 工具列表中隐藏原生 `send_emoji` |
| `max_emotion_tags` | `0` | 最多交给模型多少条描述；`0` 表示不限制 |
| `llm_model` | `"utils"` | MaiBot 中用于文本选择的模型任务名 |
| `context_message_limit` | `30` | 选择时读取多少条最近消息 |
| `semantic.enabled` | `false` | 是否优先尝试 embedding 语义匹配 |

`llm_model` 填的是 MaiBot 的模型任务名，不是模型厂商给出的模型 ID。自定义任务名也可以使用，只要它能处理普通文本生成请求。

## 它怎么选

1. 从 MaiBot 读取一次表情包列表。
2. 去掉没有描述、没有图片数据和重复描述的记录。
3. 语义模式可用时先做向量匹配，否则让文本模型从编号列表中选一个。
4. 直接从同一批记录中取出对应图片并发送。

选择和发送共用同一批记录，避免表情顺序变化后出现“选中的是 B，发出去却是 A”的情况。最近消息读取失败时会继续使用已有提示，语义匹配失败时会退回文本选择；如果最终选择或发送仍然失败，插件会停止，不会换一张继续尝试。

## 和原生表情工具的区别

| | MaiBot 原生 `send_emoji` | 本插件 |
|---|---|---|
| 选择依据 | 图片内容 | 表情描述和聊天上下文 |
| 模型要求 | 视觉模型 | 文本模型；语义模式另需 embedding |
| Planner 工具名 | `send_emoji` | `select_emoji_with_text` |
| 适合场景 | 已配置可靠的 VLM | 不想为选表情单独配置 VLM |

## 常见问题

### Planner 中看不到工具

先确认插件已经启用，并检查 MaiBot、SDK 和 Python 版本。正常情况下工具名应为 `select_emoji_with_text`，修改配置后需要让 MaiBot 完成插件重载。

### 提示“表情包库中没有可用记录”

确认 MaiBot 已经导入表情包，并且对应记录有非空描述。没有描述或图片数据读取失败的记录会被跳过。

### 提示“LLM 选择失败”或“LLM 索引解析失败”

先检查 `llm_model` 对应的模型任务是否存在。默认任务是 `utils`；如果换成自定义任务，可以在 MaiBot 日志中查看该任务有没有正常返回结果。

### 要不要打开语义匹配？

目前建议先保持 `semantic.enabled = false`。现有缓存还不能自动识别 embedding 模型或向量维度变化；如果已经启用并更换了 embedding 模型，请先停用插件，删除插件目录下的 `.cache`，再重新启用。

### 提示“发送表情包失败”

插件不会在失败后换一张继续发。请确认当前消息流仍然有效，并在 MaiBot 日志中查看 `send.emoji` 返回的错误。

## 许可证

本项目使用 GPL-3.0-or-later，完整条款见 [LICENSE](./LICENSE)。来源归属和修改说明见 [NOTICE](./NOTICE)。
