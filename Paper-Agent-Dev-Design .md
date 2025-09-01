## 项目核心定位与命名建议
+ **项目名：** Paper-Agent
+ **项目背景：**设计并落地自动化调研报告生成系统，解决领域论文调研 “耗时长、分析浅” 问题，支撑科研人员快速掌握领域现状。
+ **核心价值：** 不是简单的文献摘要器，而是一个 **“领域研究助理”** ，能自动完成“检索-阅读-分析-综合-报告”的完整工作流，提供有深度、有见解的领域综述报告。

## 第一阶段：顶层设计与技术选型 
#### 1. 定义核心功能与输出物 (MVP - 最小可行产品)
+ **输入：** 用户提供一个或多个**核心关键词**、**领域描述**、**时间范围**、**期望的论文数量**（如50篇顶会最新论文）。
+ **输出：** 一份结构化的**综述报告** (Markdown/PDF)，**至少**包含：
    - **执行摘要：** 核心发现一览。
    - **领域发展脉络：** 关键里程碑、技术演进路径。
    - **研究热点与趋势分析：** 主题模型、关键词共现分析。
    - **方法/技术对比：** 对不同流派的方法进行归纳、对比表格。
    - **权威学者与机构：** 高产作者、核心研究机构图谱。
    - **未来方向与开放问题：** 基于当前进展的合理推测。
    - **详细的参考文献列表。**

#### 2. 多智能体系统架构设计 (应对核心挑战)
采用“**AutoGen**”的多智能体协作框架，这是解决复杂链条任务的最佳实践。

+ **【大脑】主控协调智能体 (CEO Agent):**
    - **职责：** 理解用户意图， 将用户主题分解为子问题、关键词、同义词与负样本（排除项），制定总体规划，分解任务，调度下属智能体，汇总和校验最终结果。
    - **实现：** 使用最强大的LLM（如使用SiliconFlow提供商中的强大模型），通过精心设计的**系统提示词** 来扮演这个角色。
+ **【手脚】垂直领域智能体 (Worker Agents):**
    - **检索智能体 (Search Agent):**
        * 工具：调用 arXiv, Semantic Scholar, PubMed, ACL Anthology 等API。
        * 能力： 按子问题、关键词、同义词等制定检索策略，去重，筛选高质量论文【基于 DOI/标题相似度（SimHash + BM25 + embedding）去重，按时间窗、期刊等级、被引量、领域相关性打分过滤 】。
    - **阅读与摘要智能体 (Reading Agent):**
        * 工具：处理PDF，解析文本、图表、参考文献。
        * 能力：提取论文的任务、方法、数据集、指标、结论、限制条件、假设。**关键：** 使用**结构化模板**（如JSON格式）来确保信息提取的标准化，避免大模型的随意性。
    - **分析智能体 (Analysis Agent):**
        * 职责：这是**提升分析深度、避免常识偏差**的核心。
        * 能力：
            1. **关联分析：** 比较不同论文方法的异同。
            2. **趋势分析：** 分析特定技术随时间的热度变化。
            3. **辩论与校验：** 对于有矛盾的结论，可以发起“辩论”，调用多个LLM或检索更多证据来验证事实。
            4. **批判性思考：** 指出论文的局限性或潜在问题。
    - **撰写智能体 (Writing Agent):**
        * 职责： 使用结构化大纲 + 模板将分析结果生成草稿  ，撰写报告初稿。**段落内内联引用 [E12]** 对应证据表。
        * 能力：确保文风学术化、逻辑连贯、引用规范。

#### 3. 技术栈选型
+ **后端框架：** **LangChain /  LangGraph /LlamaIndex/AutoGen**。它们是构建LLM应用的事实标准，提供了强大的智能体、工具调用和文档处理能力。
+ **编程语言：** Python。
+ **LLM API:**
    - **主力：** 使用siliconflow提供商中的高性能模型。
+ **解析**：GROBID + pdfminer.six；图表 OCR 用 docTR。
+ **向量数据库：** **Chroma** ，用于存储论文嵌入，实现语义检索和关联分析。
+ **数据处理：** Pandas, NumPy 用于数据分析。
+ **前端/部署：** FastAPI (构建API)， Docker (容器化)， 后期可考虑Vue/React。

## 第二阶段：分步实现与核心挑战攻坚
#### 1. 数据采集模块
+ 集成多个学术数据源API。
+ 构建PDF解析管道，能处理不同格式的论文（PyMuPDF，OCR技术）。
+ **挑战：** 应对PDF格式不统一、图表公式提取难的问题。初期可专注于文本提取。

#### 2. 阅读与信息结构化提取 (准确性之基)
+ **这是克服大模型幻觉最关键的一步。**
+ 为“阅读智能体”设计**强约束的提示词**和**输出模板**。

**示例模板：**

    - <font style="color:rgb(82, 82, 82);background-color:rgb(250, 250, 250);">json

```plain
{
  "paper_id": "",
  "title": "",
  "authors": [],
  "publication_venue": "",
  "year": 0,
  "core_problem": "",
  "key_methodology": {
    "name": "",
    "description": "",
    "novelty": "" // 创新点
  },
  "datasets_used": [],
  "main_results": "",
  "limitations": "" // 文内提到的局限性
}
```

+ 强制LLM按照此JSON格式输出，极大减少胡言乱语，为后续分析提供高质量、结构化的数据。

#### 3. 多智能体协作逻辑实现
+ 使用AutoGen框架来编排智能体之间的工作流。
+ **典型工作流：**
    1. 用户输入关键词。
    2. **主控Agent** 激活 **检索Agent**，获取论文列表。
    3. **主控Agent** 将论文分发给多个 **阅读Agent**（并行处理以提高速度）。
    4. **主控Agent** 收集所有结构化数据，要求 **分析Agent** 进行主题聚类、趋势分析、方法对比。
    5. **主控Agent** 根据分析结果制定报告大纲，命令 **撰写Agent** 成文。
    6. **主控Agent** 最后对报告进行通读和校验，必要时发起新一轮分析。

#### 4. 提升领域分析准确性 (对抗常识偏差)
+ **检索增强生成：** 这是**黄金法则**。所有分析和撰写必须基于检索到的论文内容本身，而不是LLM的固有知识。在提示词中强制要求“根据以下论文数据进行分析...”。
+ **溯源：** 报告中的每一个观点和结论，都必须自动标注来自哪一篇论文（引用），增强可信度和可验证性。
+ **多视角校验：** 对于关键结论，可以让分析Agent从“支持”和“反对”两个角度进行论述，再由主控Agent判断哪一方证据更充分。
+ **人类反馈循环：** 设计一个界面，允许用户对报告中的错误进行标注，这些反馈可以作为未来微调LLM或优化提示词的宝贵数据。

# Paper-Agent 模块化设计与实现规划
## 项目模块划分
整个系统可以划分为以下六大核心模块，每个模块职责单一，通过清晰的接口进行通信。

| **模块名称** | **核心职责** | **关键技术/组件** |
| --- | --- | --- |
| **1. 主控协调模块 (orchestrator)** | 工作流调度、任务分解、智能体间通信 | LangGraph, LLM (强大模型) |
| **2. 数据获取模块 (data_acquire)** | 从多个源检索、下载、去重论文 | arXiv/SS API, Requests, SimHash |
| **3. 数据解析模块 (data_parse)** | 解析PDF，提取结构化文本和元数据 | GROBID, PyMuPDF, Unstructured |
| **4. 信息提取模块 (info_extract)** | 从文本中抽取标准化信息，对抗幻觉 | LLM (性价比模型), Pydantic, 提示工程 |
| **5. 分析洞察模块 (analyze_insight)** | 主题聚类、趋势分析、方法对比 | Pandas, NumPy, Sklearn, BERTopic |
| **6. 报告生成模块 (report_gen)** | 合成最终报告，管理引用 | Jinja2, Markdown, PyPDF2 |


---

## 各模块详细设计与实现
### 模块 1: 主控协调模块 (orchestrator)
这是项目的大脑，负责管理整个工作流的状态和决策。

+ **实现要点**：
    1. **使用 LangGraph 定义工作流**：创建一个有向图（Graph），节点（Node）是各个智能体或任务，边（Edge）定义执行逻辑和条件。
    2. **状态管理**：定义一个共享的`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">State**`对象，在各节点间传递。State应包含：
        * `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">user_query**`: 用户原始输入。
        * `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">search_results**`: 检索到的论文元数据列表。
        * `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">paper_contents**`: 解析后的原始文本字典（key: paper_id）。
        * `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">structured_data**`: 信息提取后的标准化JSON数据列表。
        * `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">analysis_results**`: 分析模块产生的中间结果（如聚类结果、趋势数据）。
        * `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">report_markdown**`: 最终生成的报告内容。
    3. **错误处理与重试**：为每个节点设计重试机制和降级方案（如检索失败时切换数据源）。

### 模块 2: 数据获取模块 (data_acquire)
负责根据主题获取论文列表和PDF文件。

+ **实现要点**：
    1. **多数据源代理**：编写统一的接口类，背后接入`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">arXiv**`,`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">Semantic Scholar**`,`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">PubMed**`等不同的API客户端。
    2. **检索策略**：
        * 接收主控Agent生成的**关键词列表**（包括同义词、排除词）。
        * 执行**多轮检索**，合并结果。
    3. **去重与过滤**：
        * **去重**：对检索结果按`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">标题**`进行**SimHash**计算，去除相似度极高的重复论文。
        * **过滤**：设计评分函数，根据`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">发表年份**`、`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">期刊/会议等级**`、`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">引用数**`进行初步筛选和排序。
    4. **文件下载**：异步并发下载PDF文件到本地缓存目录，避免重复下载。

### 模块 3: 数据解析模块 (data_parse)
将非结构化的PDF转换为更结构化的文本信息。

+ **实现要点**：
    1. **解析管道**：
        * 首先尝试用**PyMuPDF**快速提取文本和元数据（适用于文本型PDF）。
        * 如果失败或质量差，调用**GROBID服务**进行深度解析，特别是提取参考文献和作者信息。
        * 对于扫描件图片，集成**docTR**或**Tesseract**进行OCR。
    2. **输出标准化**：将所有解析器的输出统一为内部标准格式，包含：`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">title**`, `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">authors**`, `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">abstract**`, `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">sections**` (list of `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">{heading: '', text: ''}**`), `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">references**`。

### 模块 4: 信息提取模块 (info_extract)
**这是保证后续分析准确性的最关键模块**。利用LLM将解析后的文本抽取出高度结构化的信息。

+ **实现要点**：
    1. **提示词工程 (Prompt Engineering)**：为阅读Agent设计**严格**的提示词，强制其基于文本内容作答，并警告不要使用先验知识。

**结构化输出 (Structured Output)**：使用LangChain的`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">PydanticOutputParser**`或LlamaIndex的`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">StructuredLLMOutput**`，将LLM的输出强制约束到我们预先定义好的Pydantic模型（Schema）上，极大减少幻觉。

    2. <font style="color:rgb(82, 82, 82);background-color:rgb(250, 250, 250);">python

```plain
# 示例：使用Pydantic定义输出Schema
from pydantic import BaseModel, Field
from typing import List

class Methodology(BaseModel):
    name: str = Field(description="The name of the method")
    description: str = Field(description="A detailed description of how it works")
    novelty: str = Field(description="What is novel about this method")

class ExtractedPaperInfo(BaseModel):
    core_problem: str = Field(description="The core research problem this paper aims to solve")
    key_methodology: Methodology
    datasets_used: List[str] = Field(description="List of datasets used for experimentation")
    main_results: str = Field(description="The main findings or results reported")
    limitations: str = Field(description="The limitations discussed by the authors themselves")
```

    3. **高效处理**：对多篇论文的提取任务进行**异步并行**处理，显著提升速度。

### 模块 5: 分析洞察模块 (analyze_insight)
对提取出的结构化数据进行分析，产生洞察。

+ **实现要点**：
    1. **主题聚类**：
        * 对所有论文的`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">core_problem**`和`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">main_results**`字段生成文本嵌入（Embedding）。
        * 使用**BERTopic**或**K-Means**聚类算法自动发现研究主题。
    2. **趋势分析**：按`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">年份**`分组，统计各个主题下论文数量的变化，绘制趋势图。
    3. **方法对比**：将`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">key_methodology**`字段分组，归纳出主流方法，并对比它们的`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">main_results**`和`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">limitations**`，自动生成对比表格。
    4. **权威分析**：统计`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">authors**`和`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">institutions**`的出现频率，识别高产作者和核心机构。

### 模块 6: 报告生成模块 (report_gen)
将分析结果综合成一份易读的报告。

+ **实现要点**：
    1. **模板化**：使用**Jinja2**模板引擎来定义Markdown报告的格式。模板中包含变量占位符（如`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">{{ trend_analysis_chart }}**`）。
    2. **内容合成**：将分析模块的结果（数据框、图表、字典）填充到模板的对应位置。
    3. **引用管理**：在报告中为每一个观点自动添加引用标记`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">[^1]**`，并在文末生成规范的参考文献列表。
    4. **多格式导出**：支持将Markdown转换为PDF（通过`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">WeasyPrint**`或`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">LaTeX**`）和Word文档。

---

## 项目目录结构
一个清晰、标准的目录结构是项目成功的基础。

<font style="color:rgb(82, 82, 82);background-color:rgb(250, 250, 250);">bash

```plain
paper-agent/                  # 项目根目录
│
├── app.py                    # FastAPI应用主入口，提供Web API
├── main.py                   # 命令行应用入口
├── docker-compose.yml        # 用于部署GROBID等依赖服务
├── Dockerfile	
├── pyproject.toml            # Python项目依赖
├── .env                      # 环境变量（API Keys等）
│
├── src/                      # 主要源代码目录
│   ├── __init__.py
│   │
│   ├── agents/               # 【核心】智能体模块目录
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # 主控协调智能体 (LangGraph Graph定义地)
│   │   ├── search_agent.py   # 检索智能体
│   │   ├── reading_agent.py  # 阅读提取智能体
│   │   ├── analysis_agent.py # 分析智能体
│   │   └── writing_agent.py  # 撰写智能体
│   │
│   ├── core/                 # 核心数据模型和配置
│   │   ├── __init__.py
│   │   ├── models.py         # Pydantic数据模型 (State, PaperInfo等)
│   │   ├── config.py         # 应用配置（将.env和models.yaml中的配置结合）
│   │   ├── models.yaml       # 模型的基础配置
│   │   └── prompts.py        # 所有智能体的提示词模板
│   │
│   ├── services/             # 外部服务调用封装
│   │   ├── __init__.py
│   │   ├── arxiv_client.py   # arXiv API封装
│   │   ├── s2_client.py      # Semantic Scholar API封装
│   │   ├── grobid_client.py  # GROBID服务调用封装
│   │   └── llm_provider.py   # 统一LLM调用接口（屏蔽不同提供商差异）
│   │
│   ├── tasks/                # 具体任务实现（可供多个Agent调用）
│   │   ├── __init__.py
│   │   ├── pdf_parser.py     # PDF解析任务	
│   │   ├── paper_downloader.py # 论文下载任务
│   │   ├── deduplicator.py   # 去重任务
│   │   └── trend_analyzer.py # 趋势分析任务
│   │
│   └── utils/                # 通用工具函数
│       ├── __init__.py
│       ├── log_utils.py         # 日志配置
│       ├── cache.py          # 缓存工具
│       └── helpers.py        # 通用辅助函数
│
├── data/                     # 数据目录（.gitignore忽略）
│   ├── cache/                # 缓存下载的PDF和解析中间结果
│   ├── outputs/              # 生成的报告
│   └── tmp/                  # 临时文件
│
├── templates/                # Jinja2报告模板
│   ├── base.md.j2
│   ├── survey_report.md.j2   # 综述报告主模板
│   └── refs.bib.j2           # 参考文献模板
│
├── tests/                    # 单元测试和集成测试
│   ├── __init__.py
│   ├── test_agents/
│   ├── test_services/
│   └── test_tasks/
│
└── docs/                     # 项目文档
    ├── README.md             # 项目详细说明
    ├── DEV_SETUP.md          # 开发环境配置指南
    └── API_REFERENCE.md      # API接口文档
```

### 关键目录说明：
1. `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">src/agents/**`：这是整个项目的**智能核心**，每个Agent文件包含了该智能体的决策逻辑和与其他模块的交互。`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">orchestrator.py**`是重中之重，它用LangGraph定义了整个工作流的蓝图。
2. `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">src/core/**`：定义了项目的**数据结构**，尤其是`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">models.py**`中的Pydantic模型，确保了数据在模块间传递的规范性和类型安全。
3. `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">src/services/**`：将**所有外部依赖**（API、工具服务）封装成统一的、易于 mock 的接口，便于测试和未来更换供应商。
4. `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">src/tasks/**`：将可复用的**具体功能**（如下载、解析）拆分为独立任务，遵循单一职责原则，供多个Agent调用。
5. `**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">data/**`：所有项目运行时产生的数据都集中在这里，与代码分离，便于管理和清理。**务必在**`**<font style="color:rgb(64, 64, 64);background-color:rgb(236, 236, 236);">.gitignore**`**中忽略此目录**。

