<!-- <h1 align="center">基于多智能体和工作流的大模型的调研报告生成系统</h1> -->
<h1 align="center">Paper-Agent: 智能学术调研报告生成系统</h1>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
## 📖 简介

**Paper-Agent** 是一个面向科研人员的自动化调研报告生成系统，目标在于解决学术领域论文调研“耗时长、分析浅”的痛点。它不是简单的文献摘要工具，而是一个具备“检索-阅读-分析-综合-报告”全流程能力的智能领域研究助理，能生成有深度、有见解的领域综述报告。

**📽️ 系统演示如下**
<div align="center">
  <!-- 视频缩略图 -->
   <img width="1785" height="999" alt="视频演示缩略图1" src="https://github.com/user-attachments/assets/18f2f0bc-6d2c-4b5f-a2b9-a87d16fcd6be" />
   <img width="1779" height="1028" alt="视频演示缩略图2" src="https://github.com/user-attachments/assets/21e5dc93-1c8b-46e3-b33c-f359d94cf2db" />
   <img width="1797" height="1019" alt="视频演示缩略图3" src="https://github.com/user-attachments/assets/1e21162d-e083-40bc-93de-08302f28b08b" />
   <img width="1791" height="1020" alt="视频演示缩略图4" src="https://github.com/user-attachments/assets/77738e3d-7d80-4d8c-9ea4-61c45e3db5d6" />
  <!-- 视频链接文字 -->
   <video width="3651" controls muted>
   <source src="https://github.com/user-attachments/assets/3196bcfd-d847-4b2e-a4c4-686d5c1ebdf4" type="video/mp4">
   </video>
</div>

## ✨ 核心特性

<!-- - **多源论文检索**：支持基于关键词、领域描述、时间范围等条件，自动从 arXiv、Semantic Scholar 等数据库检索高质量论文。 -->
- 👥 **智能体协作工作流**：系统采用多智能体架构，涵盖检索、阅读、分析、写作等多个智能体，自动协作完成复杂任务。
- 📚 **自动化文献检索**：将自然语言查询转换为精确的搜索条件，获取相关学术论文
- 🔍 **结构化信息抽取**：智能阅读自动提取论文的核心问题、技术路线、实验结果、数据集、局限性等关键信息，输出标准化 JSON 结构。
- 🧠 **深度领域分析**：对文献进行聚类、对比分析，识别研究趋势和热点
- ✍️ **领域综述报告生成**：将分析结果整合成结构完整、逻辑清晰的学术报告
<!-- 自动汇总分析，生成包含执行摘要、发展脉络、研究热点、技术对比、权威机构与未来趋势等内容的结构化报告 -->
<!-- （支持 Markdown/PDF 格式）。 -->
- 🔧 **模块化设计**：各功能模块解耦，便于扩展和维护。

## 系统架构

Paper-Agent 采用模块化设计，由五个核心模块协同工作：

- **数据获取模块（search_agent）**
   - 将用户查询转换为结构化搜索条件
   - 执行论文检索并管理搜索结果
- **阅读与信息提取模块（reading_agent）**
   - 解析论文内容，提取关键信息
   - 将非结构化文本转换为结构化数据
   - 存储提取结果到向量数据库

- **数据分析模块（analyse_agent）**
   - 论文聚类与主题识别
   - 深度分析（技术趋势、方法对比等）
   - 全局综合分析与总结

- **写作模块**
   - 生成报告大纲与章节规划
   - 基于分析结果撰写各章节内容
   - 检索增强写作，补充必要资料

- **报告生成模块（report_agent）**
   - 整合章节内容，生成完整报告
   - 格式化输出符合学术规范的Markdown文档

- **主控协调模块（orchestrator）**
   - 基于LangGraph构建完整工作流
   - 协调各模块有序执行
   - 管理全局状态与错误处理


<!-- ## 技术架构 -->

<!-- - **主控协调模块（orchestrator）**：基于 LangGraph，实现智能体间的任务分配与调度。
- **数据获取模块（data_acquire）**：负责论文检索、下载与去重。
- **信息提取模块（info_extract）**：利用 LLM 及规则，提取论文结构化要素（如 core_problem、key_methodology、main_results 等）。
- **分析与综合模块（analyze & synthesize）**：聚合单篇论文信息，进行主题建模、趋势分析、机构作者统计等。
- **报告生成模块（report_generate）**：将所有分析结果以结构化报告形式输出。
- **服务与工具层（services/utils）**：封装外部 API、PDF 解析、日志等通用能力。 -->

## 工作流程

1. **输入查询**：用户提供研究主题或问题
2. **文献检索**：系统自动检索相关学术论文
3. **信息提取**：解析论文内容，提取关键信息
4. **深度分析**：聚类分析、趋势识别、方法对比
5. **内容生成**：按学术规范撰写各章节内容
6. **报告整合**：生成完整的领域综述报告

## 📂 目录结构

```text
Paper-Agents/
├── main.py                 # 应用主入口，FastAPI应用初始化
├── pyproject.toml          # Python项目配置和依赖声明
├── LICENSE                 # MIT许可证文件
├── README.md               # 项目说明文档
├── .gitignore              # Git忽略文件
├──
├── src/                    # 源代码目录
│   ├── agents/             # 智能体模块
│   │   ├── orchestrator.py         # 工作流协调器
│   │   ├── search_agent.py         # 论文检索智能体
│   │   ├── reading_agent.py        # 论文阅读智能体
│   │   ├── analyse_agent.py        # 论文分析智能体
│   │   ├── writing_agent.py        # 内容写作智能体
│   │   ├── report_agent.py         # 报告生成智能体
│   │   ├── sub_analyse_agent/      # 子分析智能体目录
│   │   └── sub_writing_agent/      # 子写作智能体目录
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
│   │   ├── deduplicator.py      # 论文去重
│   │   ├── paper_downloader.py  # 论文下载
│   │   ├── paper_filter.py      # 论文过滤
│   │   ├── paper_search.py      # 论文搜索
│   │   └── papers/              # 论文存储目录
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

<!-- 4. **查看结果**
   - 生成的报告将保存在 `output/reports/` 目录下
   - 运行日志可在 `output/logs/` 中查看 -->

4. **Web界面**
   ```bash
   cd web && npm install && npm run dev
   ```
   - 访问 http://localhost:5173 使用Web界面


## 配置说明

系统配置文件位于 `models.yaml`，可根据需求调整以下参数：
- 可选模型提供商
- 项目默认使用的模型和嵌入模型配置
- 项目模块具体使用的模型和嵌入模型配置（可选）
- 各个模型提供商的API密钥和基础URL

环境变量配置
- 在 `.env` 文件中设置API密钥

## 技术栈

### 后端
- **编程语言**: Python 3.12+
- **智能体框架**: LangGraph, AutoGen
- **Web框架**: FastAPI, Uvicorn
- **向量数据库**: ChromaDB
- **数据处理**: pyyaml, python-dotenv, tenacity
- **机器学习**: scikit-learn
- **论文检索**: arXiv API
- **网络请求**: requests, aiohttp
- **包管理**: Poetry
- **日志系统**: Python标准库logging模块 (自定义配置)
<!-- - **PDF解析**: PyMuPDF -->
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


