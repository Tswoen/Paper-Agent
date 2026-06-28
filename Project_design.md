## paper_retrieval 模块设计架构

`paper_retrieval` 是项目中负责“论文检索与结果标准化”的核心模块。它的目标不是直接暴露各个学术源站的原始返回值，而是通过统一的数据模型和服务编排层，把不同来源的检索结果转换成一致、可复用、可扩展的论文搜索能力。

### 1. 设计目标

1. 提供统一的论文检索入口，屏蔽不同数据源的 API 差异。
2. 支持单源检索与多源检索两种模式。
3. 对检索结果进行标准化、去重和基础过滤。
4. 保留来源维度的统计与错误信息，方便上层 UI 或 Agent 做诊断与展示。
5. 通过 connector 抽象便于后续快速接入新的论文数据源。

### 2. 总体分层

模块采用“服务层 + 数据模型层 + 数据源适配层”的结构。

```text
用户/Agent
   ↓
PaperSearchService
   ↓
PaperSearchConnector 抽象接口
   ↓
Arxiv / OpenAlex / Semantic Scholar 等具体 Connector
   ↓
外部学术 API
```

#### 2.0 各层角色总览

| 层级 | 角色 | 主要作用 | 对上层提供什么 | 对下层依赖什么 |
| --- | --- | --- | --- | --- |
| 用户/Agent | 调用方 | 发起论文检索请求 | 查询词、来源、过滤条件、返回数量 | 不依赖内部实现 |
| `PaperSearchService` | 编排者 / 统一入口 | 负责路由、并发、聚合、去重、错误汇总 | 标准化检索结果 `SearchResponse` | 依赖 connector 抽象和统一模型 |
| `SearchRequest` / `SearchResponse` / `PaperDocument` | 统一协议层 | 定义模块内部稳定的数据契约 | 可直接消费的统一结构 | 不依赖具体数据源字段 |
| `PaperSearchConnector` | 适配器接口 | 约束所有数据源实现统一搜索能力 | 统一的 `search(request)` 能力 | 依赖统一请求/响应模型 |
| 具体 Connector | 数据源适配器 | 对接 arXiv、OpenAlex、Semantic Scholar 等外部 API | 标准化后的论文列表 | 依赖外部 API 返回格式 |
| 外部学术 API | 数据提供者 | 提供原始论文数据 | 原始 JSON/XML 响应 | 不感知本项目内部结构 |

#### 2.1 服务层：`PaperSearchService`

服务层是模块的统一编排入口，也可以理解为“检索调度中心”。它不直接解析外部 API，而是负责把一次检索请求拆分、路由和汇总。

主要职责如下：

- 维护 connector 注册表。
- 根据 `source` 参数决定执行单源检索还是多源并发检索。
- 汇总不同来源的结果。
- 对结果进行去重和截断。
- 收集各来源的错误信息与返回数量。

服务层对上层扮演“统一出口”的角色，对下层扮演“任务分发者”的角色。
它不关心各站点 API 的细节，只处理统一后的 `SearchRequest` 和 `SearchResponse`。

#### 2.2 数据模型层：`models.py`

这一层是“数据契约层”或“统一语义层”，负责把不同来源的异构字段抽象成项目内部稳定结构。

它的角色不是做检索，也不是做网络请求，而是定义“模块里数据应该长什么样”。

这一层定义模块对外稳定的数据契约：

- `PaperDocument`：统一的论文实体模型。
- `SearchRequest`：统一的检索请求参数。
- `SearchResponse`：统一的检索响应结构。

这一层的作用是把“不同来源的异构字段”压平为“一个稳定结构”，避免上层逻辑直接依赖某个站点的字段格式。
对上层来说，它是标准结果格式；对下层来说，它是共同遵守的协议。

#### 2.3 数据源适配层：`connectors/`

connector 层是“数据源适配层”或“翻译层”，它的角色是站在外部 API 和内部模型之间做转换。

它直接对接具体外部服务，并把各自返回的数据映射到 `PaperDocument`。

当前已实现的 connector 包括：

- `ArxivPaperConnector`
- `OpenAlexPaperConnector`
- `SemanticScholarPaperConnector`

所有 connector 都继承 `PaperSearchConnector` 抽象基类，并统一实现 `search(request)`。
对上层来说，connector 提供的是一致的检索能力；对下层来说，它负责消化各自 API 的差异。

### 3. 核心数据结构

#### 3.1 `PaperDocument`

`PaperDocument` 是模块的统一论文实体，包含以下典型字段：

- `id`
- `title`
- `authors`
- `abstract`
- `year`
- `venue`
- `url`
- `pdf_url`
- `doi`
- `source`
- `metadata`

其中：

- `doi` 用于跨源去重优先匹配。
- `source` 用于标识结果来源。
- `metadata` 用于容纳各源的扩展字段，避免主结构过度膨胀。

#### 3.2 `SearchRequest`

`SearchRequest` 统一承载检索参数：

- `query`
- `source`
- `limit`
- `year_from`
- `year_to`
- `excluded_terms`

其中 `source` 允许上层指定单一来源；为空时表示启用多源联合检索。

#### 3.3 `SearchResponse`

`SearchResponse` 是对外输出的标准结果对象，除了论文列表，还保留：

- `sources_used`：本次参与检索的来源。
- `source_results`：每个来源返回的原始条数。
- `errors`：各来源的错误信息。
- `papers`：最终去重后的论文列表。
- `total`：最终可直接使用的论文数量。

这种结构适合给 UI、日志系统或 Agent 调度层做进一步处理。

### 4. 服务层工作流

#### 4.1 初始化

`PaperSearchService` 启动时会构建默认 connector 注册表，当前默认映射如下：

- `openalex` -> `OpenAlexPaperConnector`
- `semantic_scholar` -> `SemanticScholarPaperConnector`
- `semantic` -> `SemanticScholarPaperConnector`
- `arxiv` -> `ArxivPaperConnector`

这里保留了 `semantic` 和 `semantic_scholar` 两种别名，兼容不同调用方的输入习惯。

#### 4.2 单源检索

当请求指定了 `source` 时：

1. 服务层根据 source 找到对应 connector。
2. 构造 `SearchRequest` 传入 connector。
3. connector 返回统一 `PaperDocument` 列表。
4. 服务层记录来源数量，并进行一次轻量去重和截断。

#### 4.3 多源检索

当请求没有指定 `source` 时：

1. 服务层收集全部可用 connector。
2. 使用线程池并发执行多个外部 HTTP 查询。
3. 汇总每个来源的结果和异常。
4. 将所有论文合并后统一去重。
5. 按 `limit` 截断最终结果。

多源检索采用并发执行，是因为 connector 的主要耗时来自外部网络请求，线程池可以有效降低整体等待时间。

### 5. 去重策略

模块采用轻量去重策略，优先级如下：

1. 优先使用 `doi` 作为稳定键。
2. 如果没有 DOI，则退化为 `title` 的小写标准化结果。

这样可以在多源结果中尽量避免同一篇论文重复出现。当前策略是“工程可用优先”，适合论文检索场景下的快速聚合。

### 6. 各 Connector 设计

#### 6.1 抽象接口 `PaperSearchConnector`

所有数据源必须实现：

```python
search(request: SearchRequest) -> list[PaperDocument]
```

这保证服务层只依赖统一协议，而不依赖具体站点实现。

#### 6.2 `ArxivPaperConnector`

职责：

- 调用 arXiv Atom API。
- 解析 XML entry。
- 映射标题、作者、摘要、发布时间、PDF 地址等字段。
- 在 connector 内部完成年份范围和排除词过滤。

特点：

- 返回结构相对稳定，适合做基础源。
- 适合抓取预印本和开放论文。

#### 6.3 `OpenAlexPaperConnector`

职责：

- 调用 OpenAlex works API。
- 解析 JSON 结果。
- 提取作者、来源、DOI、开放 PDF 地址等信息。
- 支持基于发布时间范围的服务端过滤。

特点：

- 元数据结构较丰富。
- 适合作为开放学术元数据主来源之一。

#### 6.4 `SemanticScholarPaperConnector`

职责：

- 调用 Semantic Scholar Graph API。
- 解析结构化 JSON 返回。
- 支持 API Key 注入，以提升调用能力和稳定性。

特点：

- 字段较丰富，适合补充摘要、作者和出版信息。
- 可作为多源聚合中的重要来源。

### 7. 过滤与标准化

各 connector 在映射数据时，会执行基础标准化处理：

- 去除空字符串。
- 安全解析年份字段。
- 标准化作者列表。
- 统一 DOI 和 PDF 链接字段。
- 将源站的附加字段放入 `metadata`。

同时，connector 会应用两类基础过滤：

- 年份范围过滤：`year_from` / `year_to`
- 排除词过滤：`excluded_terms`

将这类基础过滤尽量下沉到 connector，可以减少无效结果进入上层聚合逻辑。

### 8. 并发策略

多源检索使用 `ThreadPoolExecutor` 并发执行，原因是外部检索主要是 I/O 密集型操作。

设计原则：

- 线程数量限制在合理范围内，避免过度并发。
- 每个来源独立捕获异常，不影响其他来源返回。
- 最终响应保留每个来源的错误信息，方便排查。

### 9. 错误处理

模块采用“部分成功优先”的策略：

- 某个来源失败时，不影响其他来源结果。
- 错误信息会被记录到 `SearchResponse.errors`。
- 当指定来源无效时，直接返回带错误信息的空响应。

这样上层可以在面对第三方接口波动时仍然拿到可用结果，而不是整个搜索请求失败。

### 10. 扩展方式

新增数据源时，只需要完成以下步骤：

1. 在 `connectors/` 下新增一个实现 `PaperSearchConnector` 的类。
2. 在类中实现外部 API 调用和字段映射。
3. 在 `PaperSearchService._build_default_connectors()` 中注册该 connector。
4. 如有需要，补充别名映射。
5. 增加对应测试用例。

这种设计使得“新增数据源”对业务层几乎透明。

### 11. 测试设计

当前测试重点验证服务层行为，而不是直接依赖真实外部 API：

- 单源检索能返回标准化响应。
- 多源检索能正确合并并去重。
- 无效来源会返回错误信息。

测试中通过 fake connector 模拟外部数据源，保证单元测试稳定、快速、可重复。

### 12. 小结

`paper_retrieval` 模块本质上是一个“论文检索聚合适配层”。

它通过统一模型、抽象 connector、并发聚合和轻量去重，把多个外部学术数据源组合成一个对上层友好的检索能力。这个设计的核心优势是：

- 对上层暴露统一接口。
- 方便扩展新来源。
- 可保留来源粒度的诊断信息。
- 兼顾工程可用性和实现简单性。
