项目背景
Paper-Agent 是一个面向科研人员和学生的智能论文检索与调研工具。项目基于多智能体协作架构（AutoGen + LangGraph），通过自然语言处理（NLP）、自动化搜索和知识库构建，帮助用户高效查找学术论文、分析文献内容，并进行论文调研。Paper-Agent 支持多平台集成、关键词搜索、自动分析、论文调研，提升了学术研究的效率。适用于论文写作、学术调研、科研项目管理等多种场景，是学术调研的理想助手。

环境配置
uv venv --python 3.12
.venv\Scripts\activate
<!-- uv sync 确保项目环境与锁文件保持同步 -->
uv sync 

后端启动
uv run python main.py

前端启动(在项目根目录执行即可，不需要进入front目录)
npm run front:install
npm run front:dev