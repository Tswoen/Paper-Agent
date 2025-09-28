<!-- <h1 align="center">基于多智能体和工作流的大模型的调研报告生成系统</h1> -->
<h1 align="center">Paper-Agent: 智能学术调研报告生成系统</h1>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
## 📖 简介

**Paper-Agent** 是一个面向科研人员的自动化调研报告生成系统，目标在于解决学术领域论文调研“耗时长、分析浅”的痛点。它不是简单的文献摘要工具，而是一个具备“检索-阅读-分析-综合-报告”全流程能力的智能领域研究助理，能生成有深度、有见解的领域综述报告。

Paper-Agent 是一个自动化调研报告生成系统，旨在解决领域论文调研"耗时长、分析浅"的问题，为科研人员提供一个智能的"领域研究助理"，自动完成"检索-阅读-分析-综合-报告"的完整工作流，快速生成有深度、有见解的领域综述报告。

## ✨ 核心特性

- **多源论文检索**：支持基于关键词、领域描述、时间范围等条件，自动从 arXiv、Semantic Scholar 等数据库检索高质量论文。
- **智能体协作工作流**：系统采用多智能体架构，涵盖检索、阅读、分析、写作等多个智能体，自动协作完成复杂任务。
- **结构化信息抽取**：自动提取论文的核心问题、技术路线、实验结果、数据集、局限性等关键信息，输出标准化 JSON 结构。
- **领域综述报告生成**：自动汇总分析，生成包含执行摘要、发展脉络、研究热点、技术对比、权威机构与未来趋势等内容的结构化报告（支持 Markdown/PDF 格式）。
- **模块化设计**：各功能模块解耦，便于扩展和维护。

## 核心功能

- 📚 **自动化文献检索**：将自然语言查询转换为精确的搜索条件，获取相关学术论文
- 🔍 **智能阅读分析**：自动提取论文关键信息（方法、数据集、结果等）
- 🧠 **深度领域分析**：对文献进行聚类、对比分析，识别研究趋势和热点
- ✍️ **自动报告生成**：将分析结果整合成结构完整、逻辑清晰的学术报告

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


## 技术架构

- **主控协调模块（orchestrator）**：基于 LangGraph，实现智能体间的任务分配与调度。
- **数据获取模块（data_acquire）**：负责论文检索、下载与去重。
- **信息提取模块（info_extract）**：利用 LLM 及规则，提取论文结构化要素（如 core_problem、key_methodology、main_results 等）。
- **分析与综合模块（analyze & synthesize）**：聚合单篇论文信息，进行主题建模、趋势分析、机构作者统计等。
- **报告生成模块（report_generate）**：将所有分析结果以结构化报告形式输出。
- **服务与工具层（services/utils）**：封装外部 API、PDF 解析、日志等通用能力。

## 工作流程

1. **输入查询**：用户提供研究主题或问题
2. **文献检索**：系统自动检索相关学术论文
3. **信息提取**：解析论文内容，提取关键信息
4. **深度分析**：聚类分析、趋势识别、方法对比
5. **内容生成**：按学术规范撰写各章节内容
6. **报告整合**：生成完整的领域综述报告

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

## 安装指南

```bash
# 克隆仓库
git clone https://github.com/yourusername/paper-agent.git
cd paper-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

```python
from orchestrator import PaperAgentOrchestrator

# 初始化协调器
orchestrator = PaperAgentOrchestrator()

# 运行调研流程
result = orchestrator.run(
    user_query="机器学习在医学影像诊断中的最新进展",
    max_papers=50
)

# 获取生成的报告
print(result.report)
```

## 配置说明

系统配置文件位于 `config.yaml`，可根据需求调整以下参数：

- 搜索范围与来源
- 聚类算法参数
- 模型选择与参数
- 报告格式与风格

## 技术栈

- 语言：Python 3.8+
- 框架：LangGraph
- 数据处理：Pydantic, Pandas
- 向量数据库：Chroma
- 机器学习：Scikit-learn
- 大语言模型：支持多种LLM集成


## 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

1. 提交issue报告bug或建议新功能
2. 提交pull request改进代码
3. 完善文档

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多细节。

## 许可证

本项目采用MIT许可证，详情参见 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题或建议，请联系：
- 项目主页：https://github.com/yourusername/paper-agent
- 邮箱：contact@paper-agent.com

---

⭐ 如果这个项目对你有帮助，请给我们点个星支持一下！


