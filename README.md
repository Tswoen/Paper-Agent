# Paper-Agent
<h1 align="center">基于多智能体和工作流的大模型的调研报告生成系统</h1>

## 📖 简介

**Paper-Agent** 是一个面向科研人员的自动化调研报告生成系统，目标在于解决学术领域论文调研“耗时长、分析浅”的痛点。它不是简单的文献摘要工具，而是一个具备“检索-阅读-分析-综合-报告”全流程能力的智能领域研究助理，能生成有深度、有见解的领域综述报告。

## ✨ 核心特性

- **多源论文检索**：支持基于关键词、领域描述、时间范围等条件，自动从 arXiv、Semantic Scholar 等数据库检索高质量论文。
- **智能体协作工作流**：系统采用多智能体架构，涵盖检索、阅读、分析、写作等多个智能体，自动协作完成复杂任务。
- **结构化信息抽取**：自动提取论文的核心问题、技术路线、实验结果、数据集、局限性等关键信息，输出标准化 JSON 结构。
- **领域综述报告生成**：自动汇总分析，生成包含执行摘要、发展脉络、研究热点、技术对比、权威机构与未来趋势等内容的结构化报告（支持 Markdown/PDF 格式）。
- **模块化设计**：各功能模块解耦，便于扩展和维护。

## 技术架构

- **主控协调模块（orchestrator）**：基于 LangGraph，实现智能体间的任务分配与调度。
- **数据获取模块（data_acquire）**：负责论文检索、下载与去重。
- **信息提取模块（info_extract）**：利用 LLM 及规则，提取论文结构化要素（如 core_problem、key_methodology、main_results 等）。
- **分析与综合模块（analyze & synthesize）**：聚合单篇论文信息，进行主题建模、趋势分析、机构作者统计等。
- **报告生成模块（report_generate）**：将所有分析结果以结构化报告形式输出。
- **服务与工具层（services/utils）**：封装外部 API、PDF 解析、日志等通用能力。

## 📂 目录结构

```text
paper-agent/
│
├── main.py                 # 命令行应用入口，FastAPI应用主入口
├── pyproject.toml          # Python依赖
├── .env                    # 环境变量
│
├── src/
│   ├── agents/             # 智能体模块（如 orchestrator、search_agent、reading_agent、analysis_agent、writing_agent）
│   ├── core/               # 核心数据模型与配置（如 models.py, config.py, prompts.py）
│   ├── services/           # 外部服务封装（如 arxiv_client, s2_client, grobid_client）
│   ├── tasks/              # 可复用的具体任务（如 pdf_parser, paper_downloader, deduplicator, trend_analyzer）
│   └── utils/              # 工具函数（如日志、辅助方法）
│
├── data/                   # 运行产生的数据（建议 .gitignore）
├── docs/
│   ├── README.md           # 项目详细说明
│   ├── DEV_SETUP.md        # 开发环境配置指南
│   └── API_REFERENCE.md    # API接口文档
```

## 🚀 快速开始

1. **环境准备**
   - Python 3.12+
   - 推荐使用虚拟环境
   - 安装依赖：`poetry install`
2. **配置**
   - 配置 `.env` 文件，填写各项API Key、服务地址等
   - 如需解析 PDF，确保 GROBID 服务可用
3. **运行服务**
   - 命令行模式：`poetry run python main.py`
   - 前端启动：`npm run dev`
4. **自定义任务**
   - 在 `src/agents/`、`src/tasks/` 下添加/修改智能体与任务逻辑
## 🤝 开发与贡献

- 欢迎 Issue 和 PR！
- 详细开发文档见 `docs/DEV_SETUP.md`。
- 代码遵循模块化、类型安全、易扩展的原则。

## 📜 许可协议

本项目采用 MIT License，详见 LICENSE 文件。

---

**Paper-Agent**，让学术调研更高效、更智能！