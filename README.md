<!-- <h1 align="center">基于多智能体和工作流的大模型的调研报告生成系统</h1> -->

<h1 align="center">Paper-Agent: 智能学术调研报告生成系统</h1>

<p align="center">
  语言:
  简体中文 ·
  <a href="./docs/README_en.md">English</a>
</p>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

## 📖 简介

**Paper-Agent** 是一个面向科研人员的自动化调研报告生成系统，目标在于解决学术领域论文调研“耗时长、分析浅”的痛点。它不是简单的文献摘要工具，而是一个具备“检索-阅读-分析-综合-报告”全流程能力的智能领域研究助理，能生成有深度、有见解的领域综述报告。

（趁这个暑假我将对项目进行全面重构与迭代维护，集中修复现存问题、优化整体体验。欢迎大家积极提交使用反馈、功能建议与问题 BUG，一起把项目打磨得更加完善！）

## 📸 项目预览

<summary>点击放大截图查看</summary>

| 截图1                                                                                                         | 截图2                                                                                                         | 截图3                                                                                                         |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `<img width="400" src="https://github.com/user-attachments/assets/b3617fee-ab47-4aac-9be7-0cb543fd706a" />` | `<img width="400" src="https://github.com/user-attachments/assets/a27882fb-3bd8-4f44-b18f-8161bb0d44a6" />` | `<img width="400" src="https://github.com/user-attachments/assets/18f2f0bc-6d2c-4b5f-a2b9-a87d16fcd6be" />` |
| 截图4                                                                                                         | 截图5                                                                                                         | 截图6                                                                                                         |
| `<img width="400" src="https://github.com/user-attachments/assets/21e5dc93-1c8b-46e3-b33c-f359d94cf2db" />` | `<img width="400" src="https://github.com/user-attachments/assets/1e21162d-e083-40bc-93de-08302f28b08b" />` | `<img width="400" src="https://github.com/user-attachments/assets/77738e3d-7d80-4d8c-9ea4-61c45e3db5d6" />` |

</div>

## ✨ 核心特性

- 🤖 **多智能体协作架构**：基于 AutoGen 框架，采用多智能体协作模式，涵盖检索、阅读、分析、写作等多个智能体，自动协作完成复杂任务
- 📚 **智能文献检索**：将自然语言查询转换为精确的搜索条件，支持人工审核，从 arXiv 获取相关学术论文
- 🔍 **结构化信息抽取**：智能阅读自动提取论文的核心问题、技术路线、实验结果、数据集、局限性等关键信息，输出标准化 JSON 结构
- 🧠 **深度领域分析**：通过聚类分析、深度分析、全局分析三阶段流程，识别研究趋势和热点
- ✍️ **领域综述报告生成**：将分析结果整合成结构完整、逻辑清晰的学术报告，支持 Markdown 格式输出
- 🔄 **实时流式输出**：基于 SSE（Server-Sent Events）技术，实时推送任务进度到前端
- ⚡ **并行处理优化**：支持论文并行阅读、聚类并行分析、章节并行写作，大幅提升处理效率
- 🔧 **模块化设计**：各功能模块解耦，基于 LangGraph 构建工作流，便于扩展和维护
- 💾 **向量数据库支持**：使用 ChromaDB 存储提取的论文信息，支持检索增强写作
- 👥 **用户交互审查**：在关键步骤引入人工审核，确保查询条件和生成内容符合预期

## 系统架构

**下面是简要的介绍，更多关于系统架构、节点实现、智能体协作的详细说明，请参考 [design.md](design.md) 文档。**

Paper-Agent 采用模块化设计，基于 LangGraph 构建完整的工作流，由六个核心节点协同工作：

### 核心节点

1. **search_agent_node（论文搜索节点）**

   - 使用 LLM 将用户自然语言需求转换为结构化查询条件
   - 通过用户代理（userProxyAgent）进行人工审核
   - 调用 PaperSearcher 从 arXiv 检索相关论文
   - 支持查询条件：querys、start_date、end_date
2. **reading_agent_node（论文阅读节点）**

   - 并行处理多篇论文，提取每篇论文的核心信息
   - 按照预定义模型提取：核心问题、关键方法、数据集、评估指标、主要结果、局限性、贡献
   - 将提取结果存入向量数据库，支持后续检索增强
3. **analyse_agent_node（论文分析节点）**

   - **PaperClusterAgent**：使用嵌入向量和 KMeans 算法进行论文聚类，自动确定聚类数量
   - **DeepAnalyseAgent**：对每个聚类进行深入分析，包括技术路线、方法对比、应用领域等
   - **GlobalanalyseAgent**：汇总所有聚类分析结果，生成包含六大模块的全局分析报告
4. **writing_agent_node（写作节点）**

   - **writing_director_node**：根据用户需求和全局分析，生成报告大纲并拆分为写作子任务
   - **parallel_writing_node**：并行执行所有写作子任务，使用多智能体协作完成各章节写作
   - 支持检索增强写作和质量审查
5. **report_agent_node（报告生成节点）**

   - 汇总所有写作章节，生成完整的调研报告
   - 使用 Markdown 格式输出，自动补充过渡语句
   - 流式输出，实时推送生成进度

### 子智能体架构

**写作模块子智能体**

- **writing_agent**：负责根据子任务撰写章节内容
- **retrieval_agent**：从向量数据库检索相关内容，补充写作所需资料
- **review_agent**：审查写作内容质量，通过后输出 "APPROVE" 终止子任务

**分析模块子智能体**

- **PaperClusterAgent**：论文聚类分析，生成主题描述和关键词
- **DeepAnalyseAgent**：单个聚类深度分析
- **GlobalanalyseAgent**：全局分析，生成六大模块报告

### 工作流架构

- **主控协调模块（orchestrator）**

  - 基于 LangGraph 构建完整工作流
  - 协调各节点有序执行
  - 管理全局状态与错误处理
  - 通过 SSE 实时推送任务进度到前端
- **状态管理**

  - 使用 State 管理全局状态
  - 通过队列实现前后端通信
  - 支持实时状态推送

## 工作流程

系统基于 LangGraph 构建完整的工作流，通过六个核心节点协同完成调研报告生成：

### 完整流程

1. **输入查询**：用户提供研究主题或问题
2. **论文检索**：系统自动生成查询条件，支持人工审核，从 arXiv 检索相关论文
3. **论文阅读**：并行处理多篇论文，提取核心信息并结构化
4. **深度分析**：
   - 聚类分析：按主题对论文进行分组
   - 深度分析：对每个聚类进行技术路线、方法对比等深入分析
   - 全局分析：汇总所有聚类结果，生成六大模块报告
5. **内容生成**：
   - 生成大纲：根据用户需求和全局分析，生成报告大纲
   - 任务拆分：将大纲解析为可并行执行的写作子任务
   - 并行写作：使用多智能体协作完成各章节写作
6. **报告整合**：汇总所有章节，生成完整的 Markdown 格式调研报告

### 关键特性

- **实时流式输出**：基于 SSE 技术，实时推送任务进度到前端
- **并行处理优化**：论文并行阅读、聚类并行分析、章节并行写作
- **用户交互审查**：在关键步骤引入人工审核
- **检索增强写作**：从向量数据库检索相关内容补充写作资料
- **质量审查机制**：review_agent 审查写作内容质量

## 📂 目录结构

```text
Paper-Agents/
├── main.py                 # 应用主入口，FastAPI应用初始化
├── pyproject.toml          # Python项目配置和依赖声明
├── LICENSE                 # MIT许可证文件
├── README.md               # 中文说明文档
├── design.md               # 系统设计文档
├── .gitignore              # Git忽略文件
│
├── docs/                   # 文档目录
│   └── README_en.md        # 英文说明文档
│
├── src/                    # 源代码目录
│   ├── agents/             # 智能体模块
│   │   ├── orchestrator.py         # 工作流协调器
│   │   ├── search_agent.py         # 论文检索智能体
│   │   ├── userproxy_agent.py      # 用户审查代理
│   │   ├── reading_agent.py        # 论文阅读智能体
│   │   ├── analyse_agent.py        # 论文分析智能体
│   │   ├── writing_agent.py        # 内容写作智能体
│   │   ├── report_agent.py         # 报告生成智能体
│   │   ├── sub_analyse_agent/      # 子分析智能体目录
│   │   │   ├── cluster_agent.py           # 论文聚类智能体
│   │   │   ├── deep_analyse_agent.py      # 论文深度分析智能体
│   │   │   └── global_analyse_agent.py    # 全局分析智能体
│   │   └── sub_writing_agent/      # 子写作智能体目录
│   │       ├── writing_director_agent.py    # 写作主管智能体
│   │       ├── parallel_writing_node.py     # 并行写作节点
│   │       ├── writing_agent.py             # 章节写作智能体
│   │       ├── retrieval_agent.py           # 检索增强智能体
│   │       ├── review_agent.py              # 质量审查智能体
│   │       ├── writing_chatGroup.py         # 写作协作组
│   │       └── writing_state_models.py       # 写作状态模型
│   │
│   ├── core/               # 核心模块
│   │   ├── config.py        # 配置管理
│   │   ├── model_client.py  # 模型客户端
│   │   ├── models.yaml      # 模型配置
│   │   ├── prompts.py       # 提示词模板
│   │   └── state_models.py  # 状态模型定义
│   │
│   ├── services/           # 服务层
│   │   ├── arxiv_client.py           # arXiv API客户端
│   │   ├── arxiv_fetcher.py          # arXiv论文获取器
│   │   ├── chroma_client.py          # Chroma向量数据库客户端
│   │   └── retrieval_tool.py         # 检索工具
│   │
│   ├── tasks/              # 任务模块
│   │   ├── deduplicator.py      # 论文去重（暂不支持，待完善）
│   │   ├── paper_downloader.py  # 论文下载（暂不支持，待完善）
│   │   ├── paper_filter.py      # 论文过滤（暂不支持，待完善）
│   │   ├── paper_search.py      # 论文搜索
│   │   └── papers/              # 论文存储目录（暂不支持，待完善）
│   │
│   └── utils/              # 工具函数
│       └── log_utils.py    # 日志工具
│
├── test/                   # 测试目录
│   ├── test_analyseAgent.py    # 分析智能体测试
│   ├── test_readingAgent.py    # 阅读智能体测试
│   ├── test_searchAgent.py     # 搜索智能体测试
│   ├── test_writingAgent.py    # 写作智能体测试
│   └── test_workflow.py        # 工作流测试
│
├── web/                    # 前端目录
│   ├── index.html          # 前端入口页面
│   ├── package.json        # 前端依赖配置
│   ├── src/                # 前端源代码
│   └── vite.config.js      # Vite配置
│
├── data/                   # 数据存储目录
└── output/                 # 输出目录
    └── log/                # 日志输出目录
```

## 🚀 快速开始

1. **环境准备**

   - Python 3.12+
   - 项目使用poetry 管理虚拟环境
   - 安装依赖：`poetry install`
2. **配置环境**

   - 复制 `.env.example` 为 `.env` 并填写您的API密钥
   - 修改 `models.yaml` 中的参数
3. **运行系统**

   ```bash
   poetry run python main.py
   ```
4. **Web界面**

   ```bash
   cd web && npm install && npm run dev
   ```

   - 访问 http://localhost:5173 使用Web界面

## 配置说明

### 环境变量配置

在 `.env` 文件中设置API密钥和相关配置：

```env
# 模型提供商API密钥
OPENAI_API_KEY=your_openai_api_key
# 或其他提供商的API密钥
```

### 模型配置

系统配置文件位于 `models.yaml`，可根据需求调整以下参数：

**可选模型提供商**

- OpenAI
- 其他兼容的LLM提供商

**项目默认使用的模型和嵌入模型配置**

- 默认LLM模型
- 默认嵌入模型
- 模型参数（temperature、max_tokens等）

**项目模块具体使用的模型和嵌入模型配置（可选）**

- search_agent：搜索专用模型
- reading_agent：阅读专用模型
- analyse_agent：分析专用模型
- writing_agent：写作专用模型
- report_agent：报告生成专用模型
- 各模块的嵌入模型配置

**各个模型提供商的API密钥和基础URL**

- API密钥配置
- 基础URL配置
- 其他连接参数

### 配置示例

```yaml
# models.yaml 示例
default:
  model-provider: "openai"
  model: "gpt-4"
  embedding-model: "text-embedding-3-large"
  embedding-dimension: 1024

modules:
  search_agent:
    model-provider: "openai"
    model: "gpt-3.5-turbo"
  reading_agent:
    model-provider: "openai"
    model: "gpt-4"
  analyse_agent:
    model-provider: "openai"
    model: "gpt-4"
  writing_agent:
    model-provider: "openai"
    model: "gpt-4"
  report_agent:
    model-provider: "openai"
    model: "gpt-4"

openai:
  api-key: ${OPENAI_API_KEY}
  base-url: https://api.openai.com/v1
```

## 技术栈

### 后端

- **编程语言**: Python 3.12+
- **智能体框架**:
  - AutoGen：多智能体协作框架
  - LangGraph：工作流编排框架
- **Web框架**: FastAPI, Uvicorn
- **实时通信**: SSE (Server-Sent Events)
- **向量数据库**: ChromaDB
- **数据处理**: pyyaml, python-dotenv, tenacity
- **机器学习**:
  - scikit-learn：KMeans 聚类、肘部法则
  - numpy：向量计算
- **论文检索**: arXiv API
- **网络请求**: requests, aiohttp
- **包管理**: Poetry
- **日志系统**: Python标准库logging模块 (自定义配置)

### 前端

- **框架**: Vue.js 3.4+
- **构建工具**: Vite 5.0+
- **开发工具**: @vitejs/plugin-vue

## 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

1. 提交issue报告bug或建议新功能
2. 提交pull request改进代码
3. 完善文档

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多细节。

## 许可证

本项目采用MIT许可证，详情参见 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题或建议，请通过以下方式反馈：

- **GitHub Issues**：请在项目仓库中提交Issue，这是最推荐的问题反馈方式
- 项目主页：https://github.com/Tswoen/paper-agent

---

⭐ 如果这个项目对你有帮助，请给我们点个星支持一下！

## Star 历史

[![Star History Chart](https://api.star-history.com/image?repos=Tswoen/Paper-Agent&type=date&legend=top-left)](https://www.star-history.com/?repos=Tswoen%2FPaper-Agent&type=date&logscale=&legend=top-left)
