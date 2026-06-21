`src/llm` 这个目录本质上是在做一层“统一 LLM 适配层”：上层业务只管“我要一个模型，给它消息，拿回复”，这个目录负责把不同厂商、不同协议、不同错误格式都收拢成一套统一接口。

**整体分层**
可以把它看成 5 层，从静态到运行时：

1. `registry.py`
定义“有哪些 provider，它们各自有什么静态差异”。
核心是 `ProviderSpec` 和 `match_provider()`。

2. `config.py`
定义“配置长什么样，以及怎么从配置解析出模型 preset 和 provider 连接信息”。
核心是 `ModelConfig`、`ModelPreset`、`ProviderConfig`。

3. `factory.py`
定义“如何把配置真正装配成一个可用的 provider 实例”。
核心是 `make_provider()`。

4. `base.py`
定义“所有 provider 都必须遵守的统一接口和统一返回结构”。
核心是 `LLMProvider`、`LLMResponse`、`StreamCallbacks`、`ToolCallRequest`。

5. `openai_compat.py` / `anthropic.py`
定义“具体怎么和某一类协议说话”。
它们是实际执行请求的适配器实现。

`__init__.py` 只是把这些公共入口重新导出，方便外部直接 `from llm import ...`。

**各文件分别干什么**

[registry.py](</d:/studyroom/projectspace/AIAgent_project/Paper-Agent-Projects/papers-agents/src/llm/registry.py>)
它是“路由规则表”。  
这里的 `PROVIDERS` 不是实例，而是静态描述，比如：

- 默认 `api_base`
- 默认环境变量名
- 关键字匹配规则
- 是否裁掉 `provider/model` 前缀
- 是否支持 `max_completion_tokens`

`match_provider(provider, model)` 的作用是：
如果你显式指定了 provider，就直接用；
否则根据模型名自动猜，比如 `claude-...` 归到 `anthropic`，`gpt-...` 归到 `openai`，最后兜底到 `openai_compat`。

[config.py](</d:/studyroom/projectspace/AIAgent_project/Paper-Agent-Projects/papers-agents/src/llm/config.py>)
它是“配置解析层”。

- `ProviderConfig` 管连接和鉴权：`api_key`、`api_base`、`extra_headers`、`extra_body`
- `ModelPreset` 管模型策略：`model`、`temperature`、`max_tokens`、`reasoning_effort`
- `ModelConfig` 把整个配置收起来，并提供两个关键能力：
  - `resolve_preset()`：拿到最终使用哪个 preset
  - `resolve_provider_config()`：拿到最终 provider 名称和连接配置

这个文件的价值在于把“模型策略”和“连接信息”拆开。  
同一个 OpenAI 网关可以配多个模型 preset，不会互相污染。

[base.py](</d:/studyroom/projectspace/AIAgent_project/Paper-Agent-Projects/papers-agents/src/llm/base.py>)
它是整个目录的“协议中立核心”。

最重要的是几件事：

- `LLMProvider`
  所有具体 provider 的抽象基类，规定必须实现 `chat()` 和 `chat_stream()`

- `LLMResponse`
  统一的返回结构，不管底层是 OpenAI SDK 还是 Anthropic SDK，最后都变成这个

- `ToolCallRequest`
  统一的工具调用表示

- `StreamCallbacks`
  统一流式增量回调：文本、thinking、工具调用

- `chat_with_retry()` / `chat_stream_with_retry()`
  把重试逻辑放在基类，避免每个 provider 自己写一套

- `_error_response()`
  把各种 SDK 异常、HTTP 错误统一折叠成 `LLMResponse`

所以 `base.py` 解决的是：上层不要感知“这个模型到底是谁家的”。

[factory.py](</d:/studyroom/projectspace/AIAgent_project/Paper-Agent-Projects/papers-agents/src/llm/factory.py>)
它是“装配层”。

`make_provider(config, preset_name)` 的工作流是：

1. 从 `ModelConfig` 找到 preset
2. 用 `match_provider()` 确定应该走哪个 provider
3. 解析出最终的 `api_key` / `api_base` / `extra_body`
4. 根据 `spec.backend` 实例化：
   - `OpenAICompatProvider`
   - `AnthropicProvider`
5. 生成一个 `ProviderSnapshot`

`ProviderSnapshot` 里除了 `provider` 实例，还带一个 `signature`。  
这个签名的意义是：如果配置变了，可以很容易判断“是不是该重新建 provider”。

[openai_compat.py](</d:/studyroom/projectspace/AIAgent_project/Paper-Agent-Projects/papers-agents/src/llm/openai_compat.py>)
它是 OpenAI Chat Completions 风格协议的具体实现。

它做的事主要有：

- 构造 OpenAI SDK client
- 把内部 `messages` 清洗成 OpenAI 风格
- 处理工具调用、流式输出、reasoning 内容
- 兼容不同 OpenAI-compatible 网关的小差异

它依赖 `spec`，因为同样是 OpenAI-compatible，不同网关可能有这些区别：

- 模型名前缀要不要裁
- token 参数是 `max_tokens` 还是 `max_completion_tokens`
- reasoning 模型是否禁用 `temperature`

所以它是“统一一大类兼容协议”的适配器，而不是只服务 OpenAI 官方。

[anthropic.py](</d:/studyroom/projectspace/AIAgent_project/Paper-Agent-Projects/papers-agents/src/llm/anthropic.py>)
它是 Anthropic Messages 协议的具体实现。

它和 OpenAI 版的主要区别不是“请求地址不同”，而是“消息协议完全不同”：

- `system` 不是普通消息，而是单独顶层字段
- 工具调用是 `tool_use`
- 工具返回是 `tool_result`
- `thinking` 是 Anthropic 自己的内容块类型

所以这个文件的核心工作是做协议结构转换：

- `_convert_messages()`
- `_convert_tool()`
- `_convert_tool_call()`
- `_parse_response()`

现在它也接收 `spec` 了，这让它和 `OpenAICompatProvider` 的构造方式统一起来，未来如果 `anthropic_compat` 有特殊差异，也更容易扩。

**这些文件之间怎么连起来**

运行链路基本是这样的：

1. 外部先准备配置字典
2. `ModelConfig.from_dict(...)` 解析配置
3. `make_provider(config, preset_name)` 开始装配
4. `factory.py` 调 `registry.match_provider(...)`
5. `config.py` 解析 provider 配置和环境变量
6. `factory.py` 根据 `spec.backend` 实例化具体 provider
7. 上层拿到 `ProviderSnapshot.provider`
8. 调 `provider.chat(...)` 或 `provider.chat_stream(...)`
9. 具体 provider 把统一消息格式转换成厂商协议
10. SDK 请求返回后，再解析回统一的 `LLMResponse`

也就是：

`config -> registry -> factory -> concrete provider -> unified response`

**一个很关键的设计点**
这个目录最重要的设计不是“支持多个厂商”，而是“把变化隔离开”。

变化被分散到三个位置：

- 静态差异放 `registry.py`
- 运行时配置放 `config.py`
- 协议实现差异放 `openai_compat.py` / `anthropic.py`

而上层只面对：

- `make_provider(...)`
- `provider.chat(...)`
- `provider.chat_stream(...)`
- `LLMResponse`

这就是这层架构最值钱的地方，业务代码不用知道 SDK 细节，也不用知道 OpenAI 和 Anthropic 的消息结构有多不一样。

如果你愿意，我下一步可以继续给你画一版“调用时序图”，或者直接结合项目里谁在调用 `make_provider()`，把这套架构串到真实业务入口上。