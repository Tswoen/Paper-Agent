# 项目背景
Paper-Agent 是一个面向科研人员和学生的智能论文检索与调研工具。项目基于多智能体协作架构，通过自然语言处理（NLP）、自动化搜索和知识库构建，帮助用户高效查找学术论文、分析文献内容，并进行论文调研。Paper-Agent 支持多平台集成、关键词搜索、自动分析、论文调研，提升了学术研究的效率。适用于论文写作、学术调研、科研项目管理等多种场景，是学术调研的理想助手。

# 编写代码时必须遵循的要求
1.在撰写代码时，必须给每个步骤添加详细中文注释，最好让用户能够通过注释就能理解代码。
2.写注释时，要用通俗易懂的语言来写注释，不要用什么专业术语，比如"事件流"、"桥接发布事件"等等，这些词对于一个新手来说根本搞不懂。
3.代码编写时不需要额外编写测试代码，当用户明确指出需要编写对应的测试程序才进行编写
4.在设计代码架构时，尽量不要抽象太多层，会让人看的很乱。但是为了保证以后能够增加新的功能，必要时需要模块化设计。
5.要保证整个系统的目录结构清晰干净，不要什么文件都放到一个目录下，让人看起来很乱
6.不需要兼容之前写的旧代码，要改就改的干净一些，全局都修改

# 项目结构
各个节点的Agent相关的放在src\agents中，创建对应的节点的Agent。具体仿照src\agents\searchAgent.py的实现

# 开发准则

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- Restate the request as the smallest acceptance criteria you are about to satisfy. If you cannot state it simply, you do not understand the request yet.
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
- Treat phrases like "可以", "也可以", "类似这样", or "for example" as acceptable simple directions, not permission to design a larger mechanism.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- Do not fill in imagined requirements. If you start adding aggregation, priority rules, fallback layers, protocol interpreters, or generic frameworks that were not explicitly asked for, stop and reduce the solution to the acceptance criteria.
- For small status/progress/summary changes, prefer a direct projection: read the source data, select the needed items, return the smallest useful shape. Do not rebuild an event stream or debug view unless that is the request.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.