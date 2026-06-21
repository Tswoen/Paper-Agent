# Nanobot 多厂商大模型适配架构设计文档

## 1. 文档目的

本文档基于当前 `nanobot` 项目源码实现，系统梳理其“大模型多厂商适配架构”的完整设计，目标是：

1. 解释当前项目中 LLM 适配层的真实实现方式，而不是停留在概念层。
2. 沉淀一套可在新项目中直接复刻的通用设计规范。
3. 说明不同模型提供商在鉴权、请求体、流式返回、工具调用、异常处理、模型路由等方面的差异化适配方法。

本文重点覆盖以下源码模块：

- `nanobot/providers/base.py`
- `nanobot/providers/registry.py`
- `nanobot/providers/factory.py`
- `nanobot/providers/openai_compat_provider.py`
- `nanobot/providers/anthropic_provider.py`
- `nanobot/providers/azure_openai_provider.py`
- `nanobot/providers/bedrock_provider.py`
- `nanobot/providers/openai_codex_provider.py`
- `nanobot/providers/github_copilot_provider.py`
- `nanobot/providers/fallback_provider.py`
- `nanobot/providers/openai_responses/*`
- `nanobot/config/schema.py`
- `nanobot/agent/loop.py`
- `nanobot/agent/runner.py`
- `nanobot/agent/model_presets.py`
- `nanobot/utils/llm_runtime.py`

---

## 2. 总体设计目标

Nanobot 的多厂商适配架构不是“为每个厂商写一套独立调用逻辑”，而是分成三层：

1. **统一抽象层**
   对外只暴露统一的 `LLMProvider` 能力接口，屏蔽不同厂商协议差异。

2. **Provider 元数据层**
   用 `ProviderSpec` 描述每个厂商的识别规则、鉴权方式、请求特殊性、推理参数映射规则。

3. **Provider 实现层**
   按协议族实现少量核心适配器，而不是按厂商无限拆分类：
   - OpenAI-compatible 统一适配器
   - Anthropic 原生适配器
   - Azure OpenAI 原生适配器
   - AWS Bedrock 原生适配器
   - OAuth 类 Provider 适配器

这套设计的核心思想是：

- **按协议族抽象，而不是按品牌抽象**
- **把“路由规则”与“调用实现”分离**
- **把“请求规范化”和“异常标准化”放到公共基类**
- **把大部分厂商差异收敛到元数据而不是业务代码分支**

---

## 3. 模块分层图

### 3.1 逻辑分层

```text
AgentLoop / AgentRunner
        ↓
LLMRuntime / ProviderSnapshot
        ↓
Provider Factory
        ↓
Provider Registry (ProviderSpec)
        ↓
Concrete Provider
  ├─ OpenAICompatProvider
  ├─ AnthropicProvider
  ├─ AzureOpenAIProvider
  ├─ BedrockProvider
  ├─ OpenAICodexProvider
  └─ GitHubCopilotProvider
        ↓
Upstream SDK / HTTP / SSE / Cloud API
```

### 3.2 配置到运行时装配链路

```text
config.json / env
    ↓
Config.resolve_preset()
    ↓
Config._match_provider()
    ↓
factory.make_provider()
    ↓
具体 Provider 实例
    ↓
ProviderSnapshot(provider, model, context_window_tokens, signature)
    ↓
AgentLoop / AgentRunner 发起请求
```

---

## 4. 核心抽象设计

## 4.1 `LLMProvider`：统一能力接口

文件：`nanobot/providers/base.py`

所有模型提供商最终都要实现 `LLMProvider` 抽象基类，统一能力包括：

- `chat()`
- `chat_stream()`
- `chat_with_retry()`
- `chat_stream_with_retry()`
- `get_default_model()`

这意味着上层 Agent 不关心：

- 是 OpenAI 还是 Anthropic
- 是 API Key 还是 OAuth
- 是 Responses API 还是 Chat Completions
- 是 SDK 流式还是 SSE 流式

上层只依赖“一个可聊天、可流式、可重试的 Provider”。

### 4.1.1 标准返回结构 `LLMResponse`

统一响应对象字段：

- `content`
- `tool_calls`
- `finish_reason`
- `usage`
- `retry_after`
- `reasoning_content`
- `thinking_blocks`
- `error_status_code`
- `error_kind`
- `error_type`
- `error_code`
- `error_retry_after_s`
- `error_should_retry`

这套字段是整个异常处理和 fallback 策略的基础。它把各厂商错误都标准化成统一语义，而不是把 SDK 异常直接抛给上层。

### 4.1.2 工具调用抽象 `ToolCallRequest`

统一表示模型返回的工具调用，字段包括：

- `id`
- `name`
- `arguments`
- `extra_content`
- `provider_specific_fields`
- `function_provider_specific_fields`

设计意义：

- 对内统一为 OpenAI 风格工具调用语义
- 对外保留厂商特有扩展字段，避免“通用层把特殊能力抹掉”

### 4.1.3 默认生成参数 `GenerationSettings`

Provider 自身持有：

- `temperature`
- `max_tokens`
- `reasoning_effort`

这样 `chat_with_retry()` 在调用时允许未显式传参，自动回退到 Provider 当前 generation 默认值。

---

## 4.2 `ProviderSpec`：Provider 元数据中心

文件：`nanobot/providers/registry.py`

`ProviderSpec` 是当前架构最关键的设计之一。它不是 Provider 实例，而是 Provider 的“静态协议描述”。

### 4.2.1 关键字段

#### 身份与识别

- `name`
- `display_name`
- `keywords`
- `env_key`

#### 路由与实现绑定

- `backend`
  可选值包括：
  - `openai_compat`
  - `anthropic`
  - `azure_openai`
  - `openai_codex`
  - `github_copilot`
  - `bedrock`

#### 路由辅助属性

- `is_gateway`
- `is_local`
- `is_oauth`
- `is_direct`
- `is_transcription_only`
- `detect_by_key_prefix`
- `detect_by_base_keyword`
- `default_api_base`

#### 模型名改写与参数兼容

- `strip_model_prefix`
- `strip_model_prefixes`
- `supports_max_completion_tokens`
- `model_overrides`

#### 推理/思考模式适配

- `thinking_style`
- `gateway_reasoning_style`
- `reasoning_as_content`
- `reasoning_effort_remap`
- `implicit_reasoning_models`
- `extract_thinking_blocks`
- `strip_history_reasoning_content`

#### Prompt Cache 支持

- `supports_prompt_caching`

### 4.2.2 设计价值

这意味着新增一个 OpenAI-compatible 厂商时，很多情况下不需要新写一套类，只要：

1. 在 `PROVIDERS` 中加一条 `ProviderSpec`
2. 在 `ProvidersConfig` 中补一个字段

即可完成接入。

这就是当前架构“多厂商但低类爆炸”的关键。

---

## 4.3 `ProviderSnapshot` 与 `LLMRuntime`

文件：

- `nanobot/providers/factory.py`
- `nanobot/utils/llm_runtime.py`

### 4.3.1 `ProviderSnapshot`

包含：

- `provider`
- `model`
- `context_window_tokens`
- `signature`

作用：

- 把“当前使用哪个 Provider、哪个模型、上下文窗口多大、配置签名是什么”打成一个原子快照。
- 上层运行时切换模型时，不需要关心 Provider 内部重建细节。

### 4.3.2 `signature`

`provider_signature()` 把影响运行时 Provider 链路的所有关键配置打包成一个签名，包括：

- model
- provider
- provider_name
- api_key
- api_base
- extra_headers
- extra_body
- api_type
- extra_query
- region
- profile
- max_tokens
- temperature
- reasoning_effort
- context_window_tokens
- fallback 配置

这个签名用于：

- 运行中检测配置是否变化
- 判断是否需要热刷新 Provider

### 4.3.3 `LLMRuntime`

只是对 `provider + model` 的一个轻量封装，方便工具链和子模块拿到当前运行时句柄。

---

## 5. 配置建模规范

文件：`nanobot/config/schema.py`

## 5.1 三层配置结构

### 5.1.1 Provider 配置层

`providers.<provider_name>`

负责配置：

- `api_key`
- `api_base`
- `api_type`
- `extra_headers`
- `extra_body`
- `extra_query`
- `region`
- `profile`

### 5.1.2 模型预设层

`modelPresets.<preset_name>`

负责配置：

- `model`
- `provider`
- `max_tokens`
- `context_window_tokens`
- `temperature`
- `reasoning_effort`

### 5.1.3 Agent 默认层

`agents.defaults`

负责配置：

- 当前启用的 `model_preset`
- 或 legacy 模式的 `model/provider/max_tokens/...`
- `fallback_models`
- `provider_retry_mode`

### 5.1.4 设计原则

- **Provider 层只存凭证与连接信息**
- **Preset 层只存模型与推理参数**
- **Agent 层只存当前选择和运行策略**

这样才能做到：

- 一个 Provider 复用多个模型预设
- 一个模型预设可被多个 Agent 场景复用
- fallback 可以引用完整 preset，而不是只写裸 model 名

---

## 5.2 预设解析规则

### 5.2.1 `resolve_default_preset()`

把 `agents.defaults` 的旧字段虚拟成一个隐式 preset。

### 5.2.2 `resolve_preset(name)`

规则：

- 未指定或为 `default` 时，返回隐式默认 preset
- 指定命名 preset 时，从 `model_presets` 中解析
- `model_presets.default` 被保留，不允许用户定义

### 5.2.3 fallback 规范

`agents.defaults.fallback_models` 支持两种形式：

1. 字符串：引用现有 preset 名
2. 内联对象：临时 fallback 配置

Nanobot 的实现更推荐第一种，因为 fallback 也应该是完整可复用配置，而不是散落的临时值。

---

## 6. Provider 路由规则

文件：`Config._match_provider()`，位于 `nanobot/config/schema.py`

## 6.1 路由输入

路由依据由以下几项共同决定：

- 当前生效 preset 的 `provider`
- 当前模型名 `model`
- 已配置 Provider 的 `api_key/api_base`
- 内置 `ProviderSpec` 列表顺序
- 自定义 provider 名称

## 6.2 路由优先级

### 6.2.1 显式 provider 优先

如果 preset 中 `provider != "auto"`：

1. 先找内置 spec
2. 找不到再找自定义 provider
3. 找不到则视为无效

### 6.2.2 `auto` 模式下的匹配顺序

1. **模型名前缀匹配内置 provider**
   例如 `openrouter/...`、`anthropic/...`

2. **模型名前缀匹配自定义 provider**
   例如 `companyProxy/gpt-4o-mini`

3. **按 `keywords` 做模型名关键字匹配**

4. **本地 Provider 回退**
   对 `ollama/vllm/lm_studio/...` 这类本地服务做特殊兜底

5. **网关/其他 provider fallback**
   按注册表顺序，从已配置 key 的 provider 里选

6. **最后尝试任意自定义 provider**

### 6.2.3 特殊约束

- `is_transcription_only=True` 的 Provider 不参与聊天路由
- OAuth Provider 不作为自动 fallback 候选
- 自定义 Provider 即使 `apiBase` 缺失，只要模型前缀显式命中，也不会继续误路由到别的 Provider

## 6.3 设计结论

这套路由逻辑保证：

- 明确声明时绝不猜错
- 自动模式下尽量根据模型语义推断
- 本地模型与网关模型都有兜底
- 自定义企业代理也能无侵入接入

---

## 7. Provider 工厂装配

文件：`nanobot/providers/factory.py`

## 7.1 工厂职责

`make_provider()` 不是简单 `if-else new Provider()`，而是做了四件事：

1. 解析当前生效 preset
2. 根据 config 路由出 provider_name 和 provider_config
3. 根据 `ProviderSpec.backend` 选择具体实现类
4. 如有 fallback_models，则用 `FallbackProvider` 再包一层

## 7.2 backend 到实现类映射

- `openai_compat` -> `OpenAICompatProvider`
- `anthropic` -> `AnthropicProvider`
- `azure_openai` -> `AzureOpenAIProvider`
- `bedrock` -> `BedrockProvider`
- `openai_codex` -> `OpenAICodexProvider`
- `github_copilot` -> `GitHubCopilotProvider`

## 7.3 工厂校验规则

### OpenAI-compatible

- direct 类型且没有默认 base、也没配置 `api_base` 时直接报错
- 非 local / 非 oauth / 非 direct 类型如果没 key，也报错

### Azure OpenAI

- 必须配置 `api_base`

### Bedrock

- 允许走 AWS 凭证链，不强制 `api_key`

### 自定义 Provider

- 如果名字不在内置 spec 里，会动态构造 `ProviderSpec`
- 但必须配置 `api_base`

---

## 8. 鉴权体系设计

Nanobot 支持四类鉴权路径。

## 8.1 API Key 型

适用：

- OpenAI
- OpenRouter
- DeepSeek
- DashScope
- Zhipu
- Moonshot
- MiniMax
- Mistral
- 各类 OpenAI-compatible 网关

特点：

- 主配置字段是 `api_key`
- 部分 Provider 同时写入环境变量以兼容上游 SDK
- `ProviderSpec.env_extras` 支持额外环境变量注入

## 8.2 OAuth 型

适用：

- `openai_codex`
- `github_copilot`

### 8.2.1 OpenAI Codex

文件：`nanobot/providers/openai_codex_provider.py`

特点：

- 不走普通 SDK `api_key`
- 使用 `oauth_cli_kit.get_token()` 获取 OAuth 令牌
- 直接用 `httpx` 调 Codex Responses SSE 接口

### 8.2.2 GitHub Copilot

文件：`nanobot/providers/github_copilot_provider.py`

特点：

- 先持久化 GitHub OAuth token
- 每次请求前再交换出短期 Copilot token
- Provider 本身继承 `OpenAICompatProvider`
- 只是在发请求前动态刷新 `client.api_key`

这是一种非常好的扩展方式：**复用协议适配层，只替换鉴权前置逻辑**。

## 8.3 Azure AAD / API Key 双模

文件：`nanobot/providers/azure_openai_provider.py`

规则：

- 如果 `api_key` 有值，直接用 API Key
- 如果为空，自动走 `azure.identity.aio.DefaultAzureCredential`
- scope 为 `https://cognitiveservices.azure.com/.default`

这是企业场景非常重要的设计，建议在重建时保留。

## 8.4 AWS 凭证链 / Bearer Token

文件：`nanobot/providers/bedrock_provider.py`

支持：

- `AWS_BEARER_TOKEN_BEDROCK`
- region/profile
- boto3 Session 默认凭证链

说明：

- Bedrock 的鉴权不是简单 API Key，因此必须单独做 native provider

---

## 9. OpenAI 通用抽象层设计

文件：`nanobot/providers/openai_compat_provider.py`

这是当前整个多厂商架构中最核心的适配器。

## 9.1 适配范围

该适配器统一承接：

- OpenAI
- OpenRouter
- Hugging Face Router
- Skywork
- AiHubMix
- SiliconFlow
- Novita
- VolcEngine / BytePlus
- DeepSeek
- Gemini OpenAI-compatible
- Zhipu
- DashScope
- Moonshot
- MiniMax
- Mistral
- StepFun
- Xiaomi MiMo
- LongCat
- Ant Ling
- Ollama / vLLM / LM Studio / Atomic Chat / OVMS
- Groq / Qianfan / NVIDIA NIM
- 各类自定义企业 OpenAI-compatible 代理

也就是说，绝大多数厂商差异都被压缩到了这一层。

## 9.2 请求前标准化

### 9.2.1 消息清洗 `_sanitize_empty_content()`

统一处理：

- 空字符串内容
- 空 block
- 移除 `_meta`
- `assistant + tool_calls` 时把 `content` 设为 `None`

### 9.2.2 请求级消息清洗 `_sanitize_messages()`

负责：

- 只保留 provider-safe 字段
- 工具调用 ID 标准化
- tool result ID 映射回填
- 工具参数统一序列化为 JSON object string
- 对特殊 Provider 强制把 content 压成 string
- 对严格校验 Provider 去掉历史里的 `reasoning_content`
- 最后执行 role alternation 修复

### 9.2.3 role alternation 修复 `_enforce_role_alternation()`

解决大量 OpenAI-compatible 服务的通病：

- 连续 `user` 或连续 `assistant` 被拒
- 最后一条是 `assistant` 会报错
- 历史截断后第一条非 system 若是 `assistant` 也会报错

修复策略：

- 合并连续同角色消息
- 去掉尾部 assistant prefill
- 必要时插入合成用户消息 `(conversation continued)`

这是高可复用设计，建议在新项目中原样保留。

## 9.3 模型名改写

### 9.3.1 `_request_model_name()`

用于解决网关模型名和上游模型名不一致的问题：

- `strip_model_prefix=True` 时，`provider/model` -> `model`
- `strip_model_prefixes` 时，只对指定前缀剥离

典型用途：

- GitHub Copilot
- AiHubMix
- 动态自定义 Provider

## 9.4 请求参数适配规则

### 9.4.1 `temperature`

对 `gpt-5/o1/o3/o4` 或 reasoning 打开的模型，不发送 `temperature`。

### 9.4.2 `max_tokens` vs `max_completion_tokens`

规则：

- Provider 明确声明 `supports_max_completion_tokens=True` 时使用 `max_completion_tokens`
- GPT-5 / o 系列强制使用 `max_completion_tokens`
- 其他默认用 `max_tokens`

### 9.4.3 `reasoning_effort` 统一语义层

Nanobot 把用户传入的 `reasoning_effort` 分成两层：

1. **semantic_effort**
   内部统一语义，例如 `minimal/low/medium/high/none`

2. **wire_effort**
   实际发给上游接口的值

这样就能做以下适配：

- DashScope：`minimal` -> `minimum`
- Mistral：`medium` -> `high`，`low/minimal` -> `none` 或直接省略
- Magistral：完全去掉 `reasoning_effort`
- Moonshot：有 native thinking 参数时去掉 flat `reasoning_effort`

### 9.4.4 thinking style 注入

通过 `ProviderSpec.thinking_style` 映射到不同 `extra_body`：

- `thinking_type` -> `{"thinking": {"type": "enabled/disabled"}}`
- `enable_thinking` -> `{"enable_thinking": true/false}`
- `reasoning_split` -> `{"reasoning_split": true/false}`

### 9.4.5 gateway reasoning style 注入

比如 OpenRouter 的：

- `{"reasoning": {"effort": ...}}`

### 9.4.6 `extra_body` 深度合并

Provider 内部自动注入的 thinking/body 参数，不会被用户配置完全覆盖，而是做递归 merge。

这允许用户既保留平台默认适配，又额外增加：

- `guided_json`
- `chat_template_kwargs`
- `repetition_penalty`

## 9.5 特殊 Provider 差异收敛方式

### 9.5.1 DeepSeek

- 部分历史消息必须带 `reasoning_content`
- 某些场景强制 string content

### 9.5.2 Zhipu

- 流式工具参数需要开启 `extra_body.tool_stream = true`

### 9.5.3 Moonshot / Kimi

- 某些模型强制 `temperature >= 1.0`
- 同时存在 native thinking 与 gateway reasoning 的兼容逻辑

### 9.5.4 Xiaomi MiMo

- 通过 thinking style 注入 thinking on/off
- 经 OpenRouter 路由时还要补 gateway reasoning body

### 9.5.5 Mistral

- `reasoning_effort` 词汇集与 OpenAI 不同
- 某些模型 reasoning 是隐式的，显式传参会报错
- thinking 信息藏在 content block 中，需要额外提取
- 历史消息中的 `reasoning_content` 需剥离

---

## 10. OpenAI Responses API 设计

文件：

- `nanobot/providers/openai_compat_provider.py`
- `nanobot/providers/openai_responses/converters.py`
- `nanobot/providers/openai_responses/parsing.py`

## 10.1 为什么需要单独支持 Responses API

当前实现中，Responses API 主要为以下场景服务：

- OpenAI GPT-5 / o 系列
- 显式启用 reasoning 的直连 OpenAI 请求
- GitHub Copilot 的 GPT-5 / o 系列

## 10.2 何时走 Responses API

`_should_use_responses_api()` 规则：

1. `api_type == chat_completions` -> 禁用
2. 仅 `openai` 与 `github_copilot` 支持
3. 非 GitHub Copilot 时要求直连 OpenAI base
4. reasoning 开启，或模型是 GPT-5/o 系列
5. Responses API 熔断器未打开

## 10.3 Responses API 熔断器

对每个 `model + reasoning_effort` 组合维护失败计数：

- 连续失败达到阈值后，5 分钟内不再探测
- 之后半开，允许一次探测

这样做的原因是：

- 某些网关或代理对 `/responses` 兼容不完整
- 不应每次请求都先失败一次再退回 `/chat/completions`

## 10.4 请求体转换

### 10.4.1 消息转换

`convert_messages()` 负责把 Chat Completions 风格转成 Responses 风格：

- `system` -> `instructions`
- `user` -> `input_text/input_image`
- `assistant` 文本 -> `message/output_text`
- `assistant.tool_calls` -> `function_call`
- `tool` -> `function_call_output`

### 10.4.2 工具转换

`convert_tools()` 把 OpenAI function schema 展平成 Responses `tools`。

### 10.4.3 响应解析

`parse_response_output()` 统一解析：

- 文本输出
- reasoning summary
- function_call
- usage

### 10.4.4 流式解析

`consume_sdk_stream()` / `consume_sse_with_reasoning()` 负责：

- `response.output_text.delta`
- `response.function_call_arguments.delta`
- `response.reasoning_summary_text.delta`
- `response.output_item.done`
- `response.completed`

并通过回调增量上报：

- `on_content_delta`
- `on_tool_call_delta`
- `on_reasoning_delta`

---

## 11. Anthropic 原生适配设计

文件：`nanobot/providers/anthropic_provider.py`

## 11.1 为什么单独做原生适配

原因：

- 消息协议不是 OpenAI 风格
- 工具调用不是 `tool_calls`，而是 `tool_use/tool_result` block
- 支持 extended thinking
- prompt caching 语义不同

## 11.2 消息转换

核心转换：

- `system` -> `system`
- `user` -> Anthropic content blocks
- `assistant` -> text + thinking + tool_use blocks
- `tool` -> `tool_result`

## 11.3 多模态适配

OpenAI 风格 `image_url` 会转换成 Anthropic 的：

- base64 image source
- 或 URL image source

## 11.4 工具定义转换

OpenAI：

- `function.name`
- `function.parameters`

Anthropic：

- `name`
- `input_schema`

## 11.5 thinking 适配

Anthropic 单独支持：

- `adaptive`
- `enabled + budget_tokens`

并对部分模型禁用 `temperature`。

## 11.6 流式处理

Anthropic 流式关注的不是只有 text delta，还包括：

- `thinking_delta`
- `input_json_delta`
- `content_block_start`

Nanobot 的实现特别处理了：

- thinking 期间长时间无 text token 但连接仍健康的问题
- 工具 JSON 参数的增量流式输出

---

## 12. Azure OpenAI 原生适配设计

文件：`nanobot/providers/azure_openai_provider.py`

设计特点：

- 使用 `AsyncOpenAI`
- 但目标 base URL 是 Azure `/openai/v1/`
- 核心协议使用 Responses API

### 12.1 请求体特点

- `model` 实际上对应 deployment name
- reasoning 走 `{"reasoning": {"effort": ...}}`
- reasoning 开启时不发送 `temperature`

### 12.2 鉴权双模

- API Key
- AAD `DefaultAzureCredential`

这是一个典型“协议像 OpenAI，但鉴权与 endpoint 不同”的适配器。

---

## 13. AWS Bedrock 原生适配设计

文件：`nanobot/providers/bedrock_provider.py`

## 13.1 为什么单独适配

Bedrock 使用的是 Converse / ConverseStream 协议，不是 OpenAI-compatible。

## 13.2 消息适配

转换目标包括：

- `system`
- `messages`
- `toolUse`
- `toolResult`
- `reasoningContent`
- image bytes

## 13.3 thinking 适配

Bedrock 对部分模型只支持 adaptive thinking，因此需要单独处理 `additionalModelRequestFields.thinking`。

## 13.4 工具兼容细节

即使这次请求没有显式 tools，只要历史里出现过 `toolUse/toolResult`，Bedrock 也可能要求工具配置仍然存在。

Nanobot 的做法是：

- 注入一个内部 noop tool，保证历史验证通过

这是很典型的“协议严格性兼容补丁”，在重建时很值得保留。

---

## 14. OAuth 类 Provider 设计

## 14.1 OpenAI Codex

文件：`nanobot/providers/openai_codex_provider.py`

特点：

- 不复用 OpenAI SDK 请求主链
- 直接用 `httpx` 调 Codex Responses SSE
- OAuth token 由 `oauth_cli_kit` 获取
- 流式解析复用共享 SSE parser

设计亮点：

- 虽然鉴权完全不同，但响应仍被标准化成 `LLMResponse`
- 因此上层完全无需知道这是一个特殊 Provider

## 14.2 GitHub Copilot

文件：`nanobot/providers/github_copilot_provider.py`

特点：

- 继承 `OpenAICompatProvider`
- 仅重写 token 刷新逻辑
- 请求前把 GitHub OAuth token 换成 Copilot access token

这是最推荐的扩展范式之一：

- **协议不变时，不要重复造轮子**
- **只改鉴权，不改请求解析主干**

---

## 15. 流式回调与增量解析体系

## 15.1 统一回调接口

所有 Provider 都尽量遵守这三个回调：

- `on_content_delta(text)`
- `on_thinking_delta(text)`
- `on_tool_call_delta(delta)`

其中 `on_tool_call_delta` 的通用数据结构大致包括：

- `index`
- `call_id`
- `name`
- `arguments_delta`
- 或最终 `arguments`

## 15.2 OpenAI-compatible 流式解析

来源可能有两种：

1. Chat Completions SDK stream
2. Responses API SDK stream / SSE

统一目标是抽取：

- content 增量
- reasoning 增量
- tool call 增量
- final usage

## 15.3 Anthropic 流式解析

重点事件：

- text delta
- thinking delta
- input_json_delta

## 15.4 Bedrock 流式解析

重点事件：

- `contentBlockStart`
- `contentBlockDelta`
- `contentBlockStop`
- `messageStop`
- `metadata`

## 15.5 Runner 层如何消费流

文件：`nanobot/agent/runner.py`

`AgentRunner._request_model()` 会根据能力决定：

1. **完整流式**
   直接把 delta 发给 hook

2. **进度流式**
   Provider 支持 delta，但渠道不支持完整流时，用 progress callback 增量更新

3. **非流式**
   直接 `chat_with_retry()`

同时会结合：

- `StreamingFileEditTracker`
- `IncrementalThinkExtractor`
- `on_stream_recover`

把工具调用增量、思考增量、文本增量都接入统一 Agent 体验。

---

## 16. 异常处理与重试体系

## 16.1 统一异常落地模型

所有 Provider 都应尽量把异常转成 `LLMResponse(finish_reason="error")`，并填充结构化字段：

- HTTP 状态码
- error type/code
- retry-after
- should-retry
- error_kind

## 16.2 重试入口

位于 `LLMProvider` 基类：

- `chat_with_retry()`
- `chat_stream_with_retry()`

## 16.3 瞬时错误判定

优先级：

1. `error_should_retry`
2. `error_status_code`
3. `error_kind`
4. 文本关键字兜底

429 还会继续区分：

- **可重试限流**
- **不可重试欠费/余额不足/配额耗尽**

## 16.4 `Retry-After` 提取

支持来源：

- `retry-after-ms`
- `retry-after`
- HTTP date
- 错误文本中的“retry after / try again in / wait xx sec”

## 16.5 标准重试策略

默认延迟：

- 1s
- 2s
- 4s

支持：

- `standard`
- `persistent`

`persistent` 模式会：

- 长期重试
- 但对“完全相同错误”设置停止阈值，避免无穷死循环

## 16.6 流式超时恢复规则

关键规则：

- 如果流式已经输出过正文，再遇到普通错误，不重试，防止重复输出
- 如果是 **timeout/stall**，允许新开一个流段继续重试

这是 `chat_stream_with_retry()` 与 `FallbackProvider` 中最重要的可靠性设计。

## 16.7 图像降级重试

如果请求因图片导致非瞬时错误：

- 会尝试把 `image_url` block 替换成占位文本再重试

这是一个很实用的兼容兜底。

---

## 17. 模型 fallback 设计

文件：`nanobot/providers/fallback_provider.py`

## 17.1 设计目标

当主模型失败时，自动切到备选模型，而上层 Agent 不需要写任何特殊逻辑。

## 17.2 关键设计点

- Fallback 是 **Provider 包装器**
- 每个 fallback preset 都可以对应不同 provider
- failover 是“请求级”行为，不污染长期状态

## 17.3 fallback 触发条件

通常以下错误可触发 fallback：

- timeout
- connection
- server error
- rate limit
- overloaded
- quota/balance 相关文本

但以下不会 fallback：

- auth
- permission
- content filter
- refusal
- invalid request
- context length

## 17.4 主 Provider 熔断

主模型连续失败达到阈值后会短暂熔断：

- 熔断期间跳过主模型，直接走 fallback
- 冷却结束后允许半开探测

## 17.5 上下文窗口收缩规则

`build_provider_snapshot()` 会取：

- 主模型窗口
- 所有 fallback 窗口

的最小值作为当前有效上下文窗口。

原因：

- 必须保证构造出的 prompt 对主链和所有 fallback 都可用

这个设计是工程上非常关键的一点。

---

## 18. Agent 运行时如何接入 Provider

## 18.1 `AgentLoop.from_config()`

文件：`nanobot/agent/loop.py`

启动时做的事：

1. `make_provider(config)`
2. `resolve_preset()`
3. 构建 `model_presets`
4. 构建 `preset_snapshot_loader`
5. 初始化 `AgentLoop`

## 18.2 运行时热刷新

`AgentLoop._refresh_provider_snapshot()` 每次处理消息前都会执行：

1. 读取新的 `ProviderSnapshot`
2. 比较 `signature`
3. 如有变化则调用 `_apply_provider_snapshot()`

这样可以做到：

- 配置变更后无需重启整个 Agent
- 下一轮对话自动生效

## 18.3 运行时模型切换

`set_model_preset(name)` 会：

1. 构造新 snapshot
2. 更新 `provider/model/context_window_tokens`
3. 同步给 runner / subagents / consolidator

这就是 `/model` 命令一类能力的底层基础。

---

## 19. 当前项目中的 Provider 分类总结

## 19.1 OpenAI-compatible 统一适配

适合以下情况：

- 上游接口与 OpenAI 基本兼容
- 主要差异在 base URL、headers、thinking 参数、模型名、token 限制

代表：

- OpenAI
- OpenRouter
- DeepSeek
- DashScope
- Moonshot
- Zhipu
- Mistral
- MiniMax
- Xiaomi MiMo
- 各类本地服务

## 19.2 Native 协议适配

必须单独实现：

- Anthropic
- Azure OpenAI
- Bedrock

## 19.3 OAuth 适配

适合：

- OpenAI Codex
- GitHub Copilot

## 19.4 辅助型 Provider

例如：

- AssemblyAI

它出现在统一 Provider 配置体系中，但不参与聊天模型选择。

---

## 20. 可复刻的新项目设计规范

如果你要在全新项目中完整重建这套架构，建议严格按下面的结构来落：

## 20.1 必备模块

```text
providers/
  base.py
  registry.py
  factory.py
  fallback_provider.py
  openai_compat_provider.py
  anthropic_provider.py
  azure_openai_provider.py
  bedrock_provider.py
  oauth/
    codex_provider.py
    github_copilot_provider.py
  openai_responses/
    converters.py
    parsing.py

config/
  schema.py

runtime/
  llm_runtime.py
  model_presets.py
```

## 20.2 必须保留的抽象

至少保留：

- `LLMProvider`
- `LLMResponse`
- `ToolCallRequest`
- `GenerationSettings`
- `ProviderSpec`
- `ProviderSnapshot`
- `LLMRuntime`

## 20.3 必须保留的关键能力

### 通用层

- 标准化消息清洗
- role alternation 修复
- 工具调用 ID 映射
- 统一流式回调接口
- 统一错误结构
- 重试与 retry-after 解析

### 配置层

- Provider 配置
- Model Preset 配置
- Fallback 配置
- 动态自定义 Provider 支持

### 运行时层

- Provider snapshot
- 配置签名比较
- runtime model refresh
- preset runtime switching

### Provider 层

- OpenAI-compatible 统一适配器
- Native 协议单独适配器
- OAuth provider 作为鉴权变体扩展

## 20.4 推荐实现原则

1. **新增厂商优先走 `ProviderSpec + OpenAICompatProvider`**
   只有当协议明显不同，才新增 provider class。

2. **不要让业务层直接捕获 SDK 异常**
   所有异常先沉淀成 `LLMResponse.error_*`。

3. **路由层不要直接实例化 provider**
   路由只返回“应该用谁”，实例化统一交给 factory。

4. **推理参数要做语义层与 wire 层分离**
   尤其是 `reasoning_effort`。

5. **流式解析必须支持三类增量**
   文本、思考、工具参数。

6. **fallback 必须跨 provider**
   不要把 fallback 绑定成“同厂商备份模型”。

7. **上下文窗口必须取 active chain 的最小值**
   否则主模型能过、fallback 会炸。

---

## 21. 新增 Provider 的标准接入流程

### 场景 A：新的 OpenAI-compatible 厂商

1. 在 `ProviderSpec` 中新增一条记录
2. 填写：
   - `backend="openai_compat"`
   - `default_api_base`
   - `keywords`
   - `thinking_style` / `reasoning_effort_remap` 等特殊规则
3. 在 `ProvidersConfig` 中补字段
4. 补最小测试：
   - provider 路由
   - `_build_kwargs()` 结果
   - 错误提取
   - 流式解析

### 场景 B：新的 Native 协议厂商

1. 新建 provider class
2. 实现：
   - 消息转换
   - 工具 schema 转换
   - 响应解析
   - 流式解析
   - 错误标准化
3. 在 registry 中新增 `backend`
4. 在 factory 中接入分发

### 场景 C：新的 OAuth 厂商

1. 优先判断是否能继承现有协议适配器
2. 若协议仍是 OpenAI-compatible，则继承 `OpenAICompatProvider`
3. 只实现 token 获取/刷新逻辑

---

## 22. 结论

Nanobot 当前的多厂商适配体系，本质上是一套“**统一抽象 + 元数据驱动 + 协议族适配 + 运行时快照刷新**”的设计。

其最有价值的工程思想有四点：

1. **用 `ProviderSpec` 承载大部分厂商差异，避免类爆炸。**
2. **用 `LLMProvider` 和 `LLMResponse` 抹平上层调用与异常处理差异。**
3. **用 `ProviderSnapshot + signature` 支撑运行时切换与热刷新。**
4. **用 `FallbackProvider + Retry + Responses/Chat 双通道` 提升真实生产稳定性。**

如果你要在新项目中完整复刻这套体系，最建议优先照搬的不是某个具体厂商类，而是以下四个骨架：

- `LLMProvider` 抽象
- `ProviderSpec` 注册中心
- `Factory + Snapshot` 装配链
- `OpenAICompatProvider` 统一适配主干

只要这四层骨架保持一致，后续接入新厂商的成本会非常低，且整个系统会天然具备：

- 多厂商可插拔
- 路由可控
- 异常可观测
- 流式可扩展
- fallback 可恢复

这也是当前 `nanobot` 多厂商模型适配架构的核心复用价值。
