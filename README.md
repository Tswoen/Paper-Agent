<div align="center">

# Paper-Agent · 智能论文检索与综述写作工作台

**输入一个研究主题 → 收获一份可全程追踪的领域综述**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Vue](https://img.shields.io/badge/Frontend-Vue_3-42B883?logo=vuedotjs&logoColor=white)](https://vuejs.org/)
[![LangGraph](https://img.shields.io/badge/Workflow-LangGraph-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)

</div>

---

## 🆕 2.0 版本更新

Paper-Agent 2.0 是相对旧版 1.x 的一次**全新重写**。它保留了旧版「检索 → 阅读 → 分析 → 写作」的多智能体核心思路，但在工程实现上做了全面升级：

- **前端**改为 Vue 3 + TypeScript + Vite，交互更现代、响应更快；
- **包管理**统一使用 `uv`，一条命令即可完成 Python 依赖安装；
- **论文来源**从单一 arXiv 扩展到 arXiv、OpenAlex、Semantic Scholar 三源检索；
- **会话持久化**改用 SQLite + 文件系统，浏览器刷新后历史线程不丢失；
- **实时进度**基于 SSE 推送到工作台，从检索到写作每一步都可观察。


---

## 👀 界面概览

<p align="center">
  <img src="https://github.com/user-attachments/assets/a88a2823-9724-4a7a-aee4-fafbfef068a6" width="720" alt="Paper-Agent 会话工作台" />
  <br>
  <em>输入研究主题，实时追踪检索 → 阅读 → 分析 → 大纲 → 写作全流程进度</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e312fc8b-d84e-46f5-86df-5202d2197dfe" width="720" alt="Paper-Agent 系统设置" />
  <br>
  <em>在浏览器中可视化配置模型 Provider 与 Agent 档位，一键测试连通性</em>
</p>

---

## 🎯 为什么是 Paper-Agent？

做学术调研时，你一定经历过这些：

| 场景 | 传统方式 | **Paper-Agent** |
|------|---------|-----------------|
| 初步了解陌生研究方向 | 手动搜索多个来源，逐个打开论文判断相关度，**耗时费力** | 从 arXiv / OpenAlex / Semantic Scholar 自动检索，按主题和约束去重、筛选、整理 |
| 论文阅读与资料积累 | 读完一篇记一篇笔记，资料散落在各处，**容易丢失和重复** | 先读摘要判断相关性，再按条件下载全文、解析、切分，建立本地可检索资料 |
| 撰写领域综述 | 边读边写，反复调整结构，**常常写到一半推倒重来** | 从子主题分析和全局分析生成结构化大纲，再逐节按证据写作 |
| 管理长流程任务 | 每跑一步都担心进度、状态和重启后丢失，**不敢中途停下** | 会话、运行状态、阶段产物和实时进度都在工作台可见，可随时保存与恢复 |
| 控制模型成本 | 全程用一个模型档位，**不清楚每个环节花了多少 token** | 不同阶段使用不同模型档位，并展示实际 token 用量 |

> **Paper-Agent 不只是论文摘要工具，而是一个完整的 AI 研究助理——它找得到论文、读得懂全文、理得清脉络、写得出综述。**

---

## ✨ 核心特性

| | 特性 | 一句话说明 |
|--|------|-----------|
| 🔍 | **多来源论文检索** | 内置 arXiv、OpenAlex、Semantic Scholar 连接器，统一为 `PaperDocument`，按年份、来源、数量和排除词筛选并去重评分 |
| 📖 | **从摘要到全文的渐进式阅读** | 先读摘要判断相关性，满足条件的论文走下载 → PDF 转 Markdown → 分块 → 抽取 → 写入本地向量库；依赖不可用时保存恢复现场 |
| 🔬 | **分层研究分析** | `AnalyseAgent` 先按子主题逐篇分析，再做全局综合，形成研究现状、共识、争议、空白、时间演化与展望等结构化内容 |
| ✍️ | **证据约束下的综述写作** | `WritingOutlineAgent` 生成大纲与证据映射，`WritingAgent` 逐节写作、证据不足时检索补充、完成后审查并限次修改 |
| 📡 | **实时会话工作台** | SSE 实时推送检索、阅读、分析、大纲与逐节写作进度，SQLite + 文件系统持久化，刷新后历史可恢复 |
| 🎛️ | **可视化模型配置** | 在浏览器中管理 Provider 协议、API 地址、密钥与三个 Agent 档位、embedding 参数，一键测试连通性，保存即生效 |

---

## 🔧 工作流

```mermaid
flowchart LR
    U[研究主题] --> S[SearchAgent\n生成检索计划]
    S --> R[多来源检索\narXiv / OpenAlex / Semantic Scholar]
    R --> RA[ReadAgent\n摘要阅读与相关性判断]
    RA --> FT[全文处理\n下载 / Markdown / 分块 / 向量索引]
    FT --> A[AnalyseAgent\n子主题分析与全局分析]
    A --> O[WritingOutlineAgent\n生成章节与证据映射]
    O --> W[WritingAgent\n逐节写作与审查]
    W --> P[会话产物\n综述正文与引用]
    FT -. 依赖不可用 .-> C[保存恢复现场]
    C -. 修复配置后继续 .-> RA
```

---

## 📦 技术栈

| 层级 | 技术选型 |
|------|---------|
| 运行时 | Python 3.12+、`uv`、Uvicorn |
| 后端 API | FastAPI、REST、Server-Sent Events（SSE） |
| 工作流 | LangGraph、TypedDict 共享状态 |
| Agent | Search、Read、Analyse、Writing Outline、Writing |
| LLM 适配 | OpenAI 兼容协议、Anthropic Messages 协议 |
| 论文来源 | arXiv、OpenAlex、Semantic Scholar |
| 全文处理 | `pypdf`、Markdown 转换、文本分块 |
| 向量检索 | ChromaDB |
| 会话存储 | SQLite + 本地 JSON/Markdown/向量文件 |
| 前端 | Vue 3、TypeScript、Vite、Vue Router、Lucide |

---

## 📂 项目目录

```text
Paper-Agent/
├── main.py                         # FastAPI 本地启动入口
├── pyproject.toml                  # Python 项目元数据与依赖
├── package.json                    # 根目录前端快捷命令
├── config/
│   ├── model.json                  # 本地模型配置，不提交
│   ├── settings.example.json       # 模型配置示例
│   └── system.yaml                 # 系统默认参数
├── front/                          # Vue 3 + TypeScript 前端
│   ├── src/api/                    # 会话与设置 API 客户端
│   ├── src/components/             # 工作台、状态和会话组件
│   ├── src/views/                  # 会话工作台、系统设置页
│   └── vite.config.ts              # 开发服务器、代理和端口配置
├── src/
│   ├── agents/                     # Agent 定义、模型调用和写作工具
│   ├── api/                        # FastAPI 应用与路由
│   ├── graph/                      # LangGraph 主流程和各阶段节点
│   ├── llm/                        # Provider 适配、配置解析和统一响应
│   ├── models/                     # 会话、协议和阅读领域模型
│   ├── paper_retrieval/            # 论文模型、检索服务和来源连接器
│   ├── repositories/               # SQLite、JSON、Chroma 与阶段产物持久化
│   ├── services/                   # 会话、运行、设置和工作流服务
│   └── utils/                      # 日志、缓存、全文解析和分块工具
├── data/                           # 本地数据库、论文缓存、会话和向量数据
├── logs/                           # 运行日志
└── test/                           # unittest 测试与联调辅助代码
```

---

## 🚀 快速开始

### 1. 安装项目依赖

在项目根目录执行：

```powershell
uv init
uv venv --python 3.12
uv sync
npm run front:install
```

如果你已经有可用的 Python 3.12 虚拟环境，也可以直接执行 `uv sync` 和 `npm run front:install`。

### 2. 创建本地模型配置

推荐通过 Web 工作台的「系统设置」页面修改和测试配置。也可以手动拷贝 `config/model.example.json` 为 `config/model.json`，至少确认：

1. `providers` 中存在一个可用 Provider，并填写 `api_base`；
2. `api_key` 或 `api_key_env` 能提供有效密钥；
3. `agents.default_agent` 已配置；
4. `embedding_profiles.default_embedding` 已指向可用的 embedding Provider。

### 3. 启动后端

```powershell
uv run python main.py
```

后端默认监听 `127.0.0.1:8000`，开发模式自动重载。API 文档：<http://127.0.0.1:8000/docs>

### 4. 启动前端

```powershell
npm run front:dev
```

打开 <http://127.0.0.1:5173/>，先进入「模型设置」测试模型，再进入会话工作台创建研究任务。

前端默认只监听本机，并代理 `/api`、`/webui` 请求到 `127.0.0.1:8000`。如需局域网其他设备访问：

```powershell
npm run front:dev:network
```

---

## 🔧 模型配置

系统支持多模型 Provider 配置，每个 Agent 可独立指定模型档位：

- 配置主文件：`config/model.json`（含密钥，不入库），示例见 `config/settings.example.json`，系统参数见 `config/system.yaml`

| Agent | 默认档位 | 主要职责 |
| --- | --- | --- |
| `SearchAgent` | `luna_agent` | 从研究主题生成关键词、子主题和检索约束 |
| `ReadAgent` | `default_agent` | 阅读摘要，判断相关性并整理笔记 |
| `AnalyseAgent` | `solar_agent` | 分析子主题，并综合研究现状与趋势 |
| `WritingOutlineAgent` | `default_agent` | 生成正文大纲和证据映射 |
| `WritingAgent` | `default_agent` | 逐节写作、调用资料工具和审查修改 |

没有配置 `luna_agent` / `solar_agent` 时回退到必需的 `default_agent`；`default_agent` 缺失时配置无法工作。

### Provider 后端

支持 `backend` 类型：`openai`、`openai_compat`、`anthropic`、`anthropic_compat`。示例：

```json
{
  "providers": {
    "my_provider": {
      "backend": "openai_compat",
      "api_key_env": "OPENAI_API_KEY",
      "api_base": "https://api.openai.com/v1",
      "extra_headers": {},
      "extra_body": {}
    }
  }
}
```

`api_key` 与 `api_key_env` 二选一即可。使用兼容网关时通常需要同时填写 `backend`、`api_base` 和模型名称。

### 系统参数

`config/system.yaml` 主要影响阅读节点：缓存目录、连接/下载超时、最大文件大小、文本分块、向量库路径与集合名等参数均可在此调整。

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

欢迎提交 Issue 和 Pull Request。建议贡献前先完成：

1. 在 `test/` 中补充或更新对应行为的测试；
2. 运行 `uv run python -m unittest discover -s test -v`；
3. 运行 `npm run front:build`，确保前端类型检查和构建通过；
4. 在 PR 描述中说明改动范围、配置影响和复现步骤。

项目地址：<https://github.com/Tswoen/Paper-Agent>

---

<div align="center">

**让论文检索更快，让研究脉络更清楚。**

</div>