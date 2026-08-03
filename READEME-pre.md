<div align="center">

# Paper-Agent · 智能学术调研报告生成系统

**输入一个研究主题 → 收获一份深度领域综述报告**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/Frontend-React_19-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![AutoGen](https://img.shields.io/badge/AI-AutoGen-7C3AED?logo=openai&logoColor=white)](https://microsoft.github.io/autogen)
[![LangGraph](https://img.shields.io/badge/Workflow-LangGraph-FF6B6B?)](https://langchain-ai.github.io/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?logo=open-source-initiative&logoColor=white)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?logo=github)](https://github.com/Tswoen/Paper-Agent/pulls)

<p align="center">
  <a href="docs/README_en.md">English</a> · <b>简体中文</b>
</p>

</div>

---

## 👀 界面概览

<p align="center">
  <img src="assets/paper-agent-main-UI.png" width="720" alt="Paper-Agent 主界面" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1)" />
  <br>
  <em>输入研究主题，实时追踪搜索→阅读→分析→写作全流程进度</em>
</p>

---

## 🎯 为什么是 Paper-Agent？

做学术调研时，你一定经历过这些：

| 场景 | 传统方式 | **Paper-Agent** |
|------|---------|-----------------|
| 从 0 了解一个陌生领域 | 手动搜索 → 读 20+ 篇论文 → 做笔记 → 梳理脉络，**耗时 1~2 周** | 输入研究主题，**5 分钟**拿到结构完整的领域综述 |
| 写综述报告 | 边读边写，反复调整结构，**常常写到一半推倒重来** | 自动生成大纲 → 并行写作各章节 → review 审查质量，**一次成型** |
| 挖掘研究趋势 | 靠个人经验判断，**容易遗漏重要方向** | KMeans 聚类 + 深度分析 + 全局分析，**数据驱动发现热点** |
| 和 ChatGPT 聊论文 | 单轮对话，**无法系统化组织**，输出碎片化 | 多智能体流水线，**检索→阅读→分析→写作**全流程自动化 |

> **Paper-Agent 不是论文摘要工具，而是一个完整的 AI 研究助理——它读得懂论文、理得清脉络、写得出报告。**

---

## ✨ 核心特性

| | 特性 | 一句话说明 |
|--|------|-----------|
| 🤖 | **多智能体协作流水线** | Search / Reader / Cluster / DeepAnalyse / Writer / Reviewer 各司其职，像一支研究团队自动协作 |
| 🔬 | **三阶段深度领域分析** | 聚类分析 → 每簇深度挖掘 → 六大模块全局综述，从微观到宏观全景透视 |
| ⚡ | **并行写作 + 质量审查** | 写作主管拆解大纲，多个写作小组并行输出章节，Reviewer 逐章把关 |
| 📡 | **实时流式输出** | 基于 SSE 技术，从搜索到报告每一步进度实时推送到 Web 界面 |
| 🧠 | **检索增强写作 (RAG)** | ChromaDB 向量库存储论文结构化数据，写作时自动检索相关内容辅助生成 |
| 🎛️ | **Web 可视化配置** | 在浏览器中管理多模型 Provider、为每个 Agent 单独指定模型、一键测试连通性 |

---

## 📦 技术栈

| 层级 | 技术选型 |
|------|---------|
| **AI 框架** | AutoGen, LangGraph, LangChain |
| **后端** | Python 3.12+, FastAPI, Uvicorn, SSE |
| **向量数据库** | ChromaDB |
| **机器学习** | scikit-learn (KMeans, Elbow Method), NumPy |
| **前端** | React 19, TypeScript, Vite 7 |
| **论文检索** | arXiv API, aiohttp |
| **PDF 解析** | PyMuPDF (fitz) |
| **包管理** | Poetry (Python) / npm (Web) |
| **LLM 兼容** | OpenAI, SiliconFlow, DashScope, Ark 及兼容接口 |

> 🧠 详细系统设计文档请参考 [`design.md`](design.md)

### 📂 项目目录

```text
Paper-Agent/
├── main.py                 # 应用主入口，FastAPI 应用初始化
├── pyproject.toml          # Python 项目配置和依赖声明
├── LICENSE                 # MIT 许可证文件
├── README.md               # 中文说明文档
├── design.md               # 系统设计文档
├── .gitignore              # Git 忽略文件
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
│   │       └── writing_state_models.py      # 写作状态模型
│   │
│   ├── core/               # 核心模块
│   │   ├── config.py        # 配置管理
│   │   ├── config_router.py # 系统配置接口
│   │   ├── model_client.py  # 模型客户端
│   │   ├── models.yaml      # 模型配置
│   │   ├── system_params.yaml # 系统参数配置
│   │   ├── prompts.py       # 提示词模板
│   │   └── state_models.py  # 状态模型定义
│   │
│   ├── services/           # 服务层
│   │   ├── chroma_client.py          # Chroma 向量数据库客户端
│   │   └── retrieval_tool.py         # 检索工具
│   │
│   ├── knowledge/          # 知识库模块
│   │   ├── knowledge_router.py      # 知识库 API 接口
│   │   └── knowledge/               # 知识库实现
│   │
│   ├── tasks/              # 任务模块
│   │   └── paper_search.py # 论文搜索
│   │
│   ├── plugins/            # 插件模块
│   │
│   └── utils/              # 工具函数
│       └── log_utils.py    # 日志工具
│
├── web/                    # 前端目录
│   ├── index.html          # 前端入口页面
│   ├── package.json        # 前端依赖配置
│   ├── vite.config.ts      # Vite 配置
│   ├── tsconfig.json       # TypeScript 配置
│   └── src/                # 前端源代码
│       ├── main.tsx        # 应用入口
│       ├── App.tsx         # 根组件
│       ├── styles.css      # 全局样式
│       ├── api/            # API 客户端
│       │   ├── config.ts
│       │   ├── knowledge.ts
│       │   └── knowledge.test.ts
│       ├── features/       # 功能模块
│       │   ├── research/   # 研究查询页面
│       │   ├── config/     # 系统配置页面
│       │   ├── knowledge/  # 知识库管理页面
│       │   └── history/    # 历史记录页面
│       └── test/           # 前端测试
│           └── setup.ts
│
├── test/                   # 测试目录
│   ├── test_analyseAgent.py    # 分析智能体测试
│   ├── test_readingAgent.py    # 阅读智能体测试
│   ├── test_searchAgent.py     # 搜索智能体测试
│   ├── test_writingAgent.py    # 写作智能体测试
│   └── test_workflow.py        # 工作流测试
│
├── data/                   # 数据存储目录
└── output/                 # 输出目录
    └── log/                # 日志输出目录
```

---

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/Tswoen/Paper-Agent.git && cd Paper-Agent

# 2. 安装 Python 依赖
poetry install

# 3. 配置环境变量
cp .env.example .env   # 填入你的 API Key

# 4. 启动后端（默认 :8000）
poetry run python main.py

# 5. 启动前端（默认 :5173，新开终端）
cd web && npm install && npm run dev
```

打开浏览器访问 **http://localhost:5173**，输入你的研究主题即可开始体验。

> 💡 API Key 也可以通过 Web 界面在「系统配置」页面中填写和保存，无需手动编辑 `.env` 文件。

### 🔧 配置说明

系统支持多模型 Provider 配置，每个 Agent 可独立指定模型：

- 配置主文件：`src/core/models.yaml`
- 支持通过 Web 界面可视化编辑（推荐方式）
- 可选 Provider：OpenAI / SiliconFlow / DashScope / Ark / 任意 OpenAI 兼容服务
- 可单独配置：搜索、阅读、分析、写作、报告生成等 Agent 的 LLM & 嵌入模型
- 支持一键测试模型连通性

---

## 💬 交流社群

加入 Paper-Agent 用户交流群，获取最新动态、使用技巧与技术讨论：

<p align="center">
  <img src="assets/paper-agent-QQ.jpg" width="280" alt="QQ 交流群二维码" />
  <br>
  <em>（QQ 群号：340020097）</em>
</p>

---

## ❤️ 特别致谢

感谢 **@GreatZack** 对 Paper-Agent 的持续投入与核心贡献：

<p align="center">
  <a href="https://github.com/GreatZack">
    <img src="https://github.com/GreatZack.png" width="80" height="80" style="border-radius:50%" alt="GreatZack" />
  </a>
  <br>
  <strong><a href="https://github.com/GreatZack">@GreatZack</a></strong>
</p>

---

## 🤝 参与贡献

欢迎各种形式的贡献！包括但不限于：

- 提交 [Issue](https://github.com/Tswoen/Paper-Agent/issues) 报告 BUG 或建议新功能
- 提交 [Pull Request](https://github.com/Tswoen/Paper-Agent/pulls) 改进代码
- 完善文档或分享使用案例

---

## 📄 开源协议

[MIT](LICENSE) © 2024 Tswoen

---

<div align="center">

**如果 Paper-Agent 对你的研究有帮助，请为我们点一个 ⭐**

[![Star History Chart](https://api.star-history.com/image?repos=Tswoen/Paper-Agent&type=Date)](https://star-history.com/#Tswoen/Paper-Agent&Date)

</div>