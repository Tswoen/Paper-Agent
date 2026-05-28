<!-- <h1 align="center">基于多智能体和工作流的大模型的调研报告生成系统</h1> -->

<h1 align="center">Paper-Agent: Intelligent Academic Survey Report Generation System</h1>

<p align="center">
  Languages:
  <a href="../README.md">简体中文</a> ·
  English
</p>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

## 📖 Introduction

**Paper-Agent** is an automated survey report generation system for researchers, designed to address the pain points of "time-consuming and shallow analysis" in academic paper research. It is not a simple literature summarization tool, but an intelligent domain research assistant with full-process capabilities of "retrieval-reading-analysis-synthesis-report" that can generate in-depth and insightful domain survey reports.

（I will fully refactor, iterate and maintain this project during the summer vacation to fix existing issues and optimize the overall experience. Feel free to submit your usage feedback, feature requests and bug reports. Let's work together to make this project better!）

## 📸 Project Preview

<summary>Click to enlarge screenshots</summary>

| Screenshot 1                                                                                                  | Screenshot 2                                                                                                  | Screenshot 3                                                                                                  |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| <img width="400" src="https://github.com/user-attachments/assets/b3617fee-ab47-4aac-9be7-0cb543fd706a" /> | <img width="400" src="https://github.com/user-attachments/assets/a27882fb-3bd8-4f44-b18f-8161bb0d44a6" /> | <img width="400" src="https://github.com/user-attachments/assets/18f2f0bc-6d2c-4b5f-a2b9-a87d16fcd6be" /> |
| Screenshot 4                                                                                                  | Screenshot 5                                                                                                  | Screenshot 6                                                                                                  |
| <img width="400" src="https://github.com/user-attachments/assets/21e5dc93-1c8b-46e3-b33c-f359d94cf2db" /> | <img width="400" src="https://github.com/user-attachments/assets/1e21162d-e083-40bc-93de-08302f28b08b" /> | <img width="400" src="https://github.com/user-attachments/assets/77738e3d-7d80-4d8c-9ea4-61c45e3db5d6" /> |

</div>

## ✨ Core Features

- 🤖 **Multi-Agent Collaboration Architecture**: Based on the AutoGen framework, adopting a multi-agent collaboration model covering retrieval, reading, analysis, writing, and other agents to automatically collaborate on complex tasks
- 📚 **Intelligent Literature Retrieval**: Converts natural language queries into precise search conditions with manual review support, retrieves relevant academic papers from arXiv
- 🔍 **Structured Information Extraction**: Intelligent reading extracts core problems, technical approaches, experimental results, datasets, limitations, and other key information from papers, outputting standardized JSON structures
- 🧠 **In-Depth Domain Analysis**: Through three-stage processes of cluster analysis, deep analysis, and global analysis, identifies research trends and emerging topics
- ✍️ **Domain Survey Report Generation**: Integrates analysis results into academically structured reports with clear logic, supporting Markdown format output
- 🔄 **Real-time Streaming Output**: Based on SSE (Server-Sent Events) technology, pushes task progress to the frontend in real-time
- ⚡ **Parallel Processing Optimization**: Supports parallel paper reading, parallel cluster analysis, and parallel chapter writing, significantly improving processing efficiency
- 🔧 **Modular Design**: Decoupled functional modules, built on LangGraph for workflow, easy to extend and maintain
- 💾 **Vector Database Support**: Uses ChromaDB to store extracted paper information, supporting retrieval-augmented writing
- 👥 **User Interaction Review**: Introduces manual review at key steps to ensure query conditions and generated content meet expectations

## System Architecture

**Here is a brief introduction. For more detailed information about system architecture, node implementation, and agent collaboration, please refer to the [design.md](../design.md) document.**

Paper-Agent adopts a modular design, builds a complete workflow based on LangGraph, and works through six core nodes:

### Core Nodes

1. **search_agent_node (Paper Search Node)**

   - Uses LLM to convert user natural language requirements into structured query conditions
   - Performs manual review through user proxy (userProxyAgent)
   - Calls PaperSearcher to retrieve relevant papers from arXiv
   - Supports query conditions: querys, start_date, end_date
2. **reading_agent_node (Paper Reading Node)**

   - Processes multiple papers in parallel, extracting core information from each paper
   - Extracts according to predefined models: core problem, key methods, datasets, evaluation metrics, main results, limitations, contributions
   - Stores extracted results in vector database for subsequent retrieval augmentation
3. **analyse_agent_node (Paper Analysis Node)**

   - **PaperClusterAgent**: Uses embedding vectors and KMeans algorithm for paper clustering, automatically determining cluster count
   - **DeepAnalyseAgent**: Performs in-depth analysis on each cluster, including technical approaches, method comparisons, application domains, etc.
   - **GlobalanalyseAgent**: Summarizes all cluster analysis results to generate a global analysis report containing six major modules
4. **writing_agent_node (Writing Node)**

   - **writing_director_node**: Generates report outline and splits it into writing sub-tasks based on user requirements and global analysis
   - **parallel_writing_node**: Executes all writing sub-tasks in parallel, using multi-agent collaboration to complete chapter writing
   - Supports retrieval-augmented writing and quality review
5. **report_agent_node (Report Generation Node)**

   - Summarizes all written chapters to generate a complete survey report
   - Outputs in Markdown format, automatically adding transitional sentences
   - Streaming output, pushing generation progress in real-time

### Sub-Agent Architecture

**Writing Module Sub-Agents**

- **writing_agent**: Responsible for writing chapter content based on sub-tasks
- **retrieval_agent**: Retrieves relevant content from vector database to supplement writing materials
- **review_agent**: Reviews writing content quality, outputs "APPROVE" to terminate sub-task upon approval

**Analysis Module Sub-Agents**

- **PaperClusterAgent**: Paper cluster analysis, generates topic
- **DeepAnalyseAgent**: In-depth analysis of single descriptions and keywords
  clusters
- **GlobalanalyseAgent**: Global analysis, generates six-module report

### Workflow Architecture

- **Orchestrator Module**

  - Builds complete workflow based on LangGraph
  - Coordinates orderly execution of nodes
  - Manages global state and error handling
  - Pushes task progress to frontend in real-time via SSE
- **State Management**

  - Uses State to manage global state
  - Implements frontend-backend communication through queues
  - Supports real-time state push

## Workflow

The system builds a complete workflow based on LangGraph and completes survey report generation through six core nodes:

### Complete Process

1. **Input Query**: User provides research topic or question
2. **Paper Retrieval**: System automatically generates query conditions with manual review support, retrieves relevant papers from arXiv
3. **Paper Reading**: Processes multiple papers in parallel, extracts and structures core information
4. **In-Depth Analysis**:
   - Cluster Analysis: Groups papers by topic
   - Deep Analysis: Performs in-depth analysis on each cluster including technical approaches, method comparisons, etc.
   - Global Analysis: Summarizes all cluster results to generate six-module report
5. **Content Generation**:
   - Generate Outline: Generates report outline based on user requirements and global analysis
   - Task Splitting: Parses outline into parallel executable writing sub-tasks
   - Parallel Writing: Uses multi-agent collaboration to complete chapter writing
6. **Report Integration**: Summarizes all chapters to generate complete Markdown format survey report

### Key Features

- **Real-time Streaming Output**: Based on SSE technology, pushes task progress to frontend in real-time
- **Parallel Processing Optimization**: Parallel paper reading, parallel cluster analysis, parallel chapter writing
- **User Interaction Review**: Introduces manual review at key steps
- **Retrieval-Augmented Writing**: Retrieves relevant content from vector database to supplement writing materials
- **Quality Review Mechanism**: review_agent reviews writing content quality

<!-- ## 📂 Directory Structure

```text
Paper-Agents/
├── main.py                 # Application main entry, FastAPI application initialization
├── pyproject.toml          # Python project configuration and dependency declaration
├── LICENSE                 # MIT license file
├── README.md               # Project documentation
├── .gitignore              # Git ignore file
│
├── src/                    # Source code directory
│   ├── agents/             # Agent module
│   │   ├── orchestrator.py         # Workflow orchestrator
│   │   ├── search_agent.py         # Paper search agent
│   │   ├── userproxy_agent.py      # User review agent
│   │   ├── reading_agent.py        # Paper reading agent
│   │   ├── analyse_agent.py        # Paper analysis agent
│   │   ├── writing_agent.py        # Content writing agent
│   │   ├── report_agent.py         # Report generation agent
│   │   ├── sub_analyse_agent/      # Sub-analysis agent directory
│   │   │   ├── cluster_agent.py           # Paper cluster agent
│   │   │   ├── deep_analyse_agent.py      # Paper deep analysis agent
│   │   │   └── global_analyse_agent.py    # Global analysis agent
│   │   └── sub_writing_agent/      # Sub-writing agent directory
│   │       ├── writing_director_agent.py    # Writing director agent
│   │       ├── parallel_writing_node.py     # Parallel writing node
│   │       ├── writing_agent.py             # Chapter writing agent
│   │       ├── retrieval_agent.py           # Retrieval augmentation agent
│   │       ├── review_agent.py              # Quality review agent
│   │       ├── writing_chatGroup.py         # Writing collaboration group
│   │       └── writing_state_models.py       # Writing state models
│   │
│   ├── core/               # Core module
│   │   ├── config.py        # Configuration management
│   │   ├── model_client.py  # Model client
│   │   ├── models.yaml      # Model configuration
│   │   ├── prompts.py       # Prompt templates
│   │   └── state_models.py  # State model definitions
│   │
│   ├── services/           # Service layer
│   │   ├── arxiv_client.py           # arXiv API client
│   │   ├── arxiv_fetcher.py          # arXiv paper fetcher
│   │   ├── chroma_client.py          # Chroma vector database client
│   │   └── retrieval_tool.py         # Retrieval tool
│   │
│   ├── tasks/              # Task module
│   │   ├── deduplicator.py      # Paper deduplication (not supported yet, to be completed)
│   │   ├── paper_downloader.py  # Paper download (not supported yet, to be completed)
│   │   ├── paper_filter.py      # Paper filtering (not supported yet, to be completed)
│   │   ├── paper_search.py      # Paper search
│   │   └── papers/              # Paper storage directory (not supported yet, to be completed)
│   │
│   └── utils/              # Utility functions
│       └── log_utils.py    # Logging utilities
│
├── test/                   # Test directory
│   ├── test_analyseAgent.py    # Analysis agent test
│   ├── test_readingAgent.py    # Reading agent test
│   ├── test_searchAgent.py     # Search agent test
│   ├── test_writingAgent.py    # Writing agent test
│   └── test_workflow.py        # Workflow test
│
├── web/                    # Frontend directory
│   ├── index.html          # Frontend entry page
│   ├── package.json        # Frontend dependency configuration
│   ├── src/                # Frontend source code
│   └── vite.config.js      # Vite configuration
│
├── data/                   # Data storage directory
└── output/                 # Output directory
    └── log/                # Log output directory
``` -->

## 📂 Directory Structure

```text
Paper-Agents/
├── main.py                 # Application main entry, FastAPI application initialization
├── pyproject.toml          # Python project configuration and dependency declaration
├── LICENSE                 # MIT license file
├── README.md               # Chinese documentation
├── design.md               # System design document
├── .gitignore              # Git ignore file
│
├── docs/                   # Documentation directory
│   └── README_en.md        # English documentation
│
├── src/                    # Source code directory
│   ├── agents/             # Agent module
│   │   ├── orchestrator.py         # Workflow orchestrator
│   │   ├── search_agent.py         # Paper search agent
│   │   ├── userproxy_agent.py      # User review agent
│   │   ├── reading_agent.py        # Paper reading agent
│   │   ├── analyse_agent.py        # Paper analysis agent
│   │   ├── writing_agent.py        # Content writing agent
│   │   ├── report_agent.py         # Report generation agent
│   │   ├── sub_analyse_agent/      # Sub-analysis agent directory
│   │   │   ├── cluster_agent.py           # Paper cluster agent
│   │   │   ├── deep_analyse_agent.py      # Paper deep analysis agent
│   │   │   └── global_analyse_agent.py    # Global analysis agent
│   │   └── sub_writing_agent/      # Sub-writing agent directory
│   │       ├── writing_director_agent.py    # Writing director agent
│   │       ├── parallel_writing_node.py     # Parallel writing node
│   │       ├── writing_agent.py             # Chapter writing agent
│   │       ├── retrieval_agent.py           # Retrieval augmentation agent
│   │       ├── review_agent.py              # Quality review agent
│   │       ├── writing_chatGroup.py         # Writing collaboration group
│   │       └── writing_state_models.py      # Writing state models
│   │
│   ├── core/               # Core module
│   │   ├── config.py        # Configuration management
│   │   ├── model_client.py  # Model client
│   │   ├── models.yaml      # Model configuration
│   │   ├── prompts.py       # Prompt templates
│   │   └── state_models.py  # State model definitions
│   │
│   ├── services/           # Service layer
│   │   ├── arxiv_client.py           # arXiv API client
│   │   ├── arxiv_fetcher.py          # arXiv paper fetcher
│   │   ├── chroma_client.py          # Chroma vector database client
│   │   └── retrieval_tool.py         # Retrieval tool
│   │
│   ├── tasks/              # Task module
│   │   ├── deduplicator.py      # Paper deduplication (planned)
│   │   ├── paper_downloader.py  # Paper download (planned)
│   │   ├── paper_filter.py      # Paper filtering (planned)
│   │   ├── paper_search.py      # Paper search
│   │   └── papers/              # Paper storage directory (planned)
│   │
│   └── utils/              # Utility functions
│       └── log_utils.py    # Logging utilities
│
├── test/                   # Test directory
│   ├── test_analyseAgent.py    # Analysis agent test
│   ├── test_readingAgent.py    # Reading agent test
│   ├── test_searchAgent.py     # Search agent test
│   ├── test_writingAgent.py    # Writing agent test
│   └── test_workflow.py        # Workflow test
│
├── web/                    # Frontend directory
│   ├── index.html          # Frontend entry page
│   ├── package.json        # Frontend dependency configuration
│   ├── src/                # Frontend source code
│   └── vite.config.js      # Vite configuration
│
├── data/                   # Data storage directory
└── output/                 # Output directory
    └── log/                # Log output directory


## 🚀 Quick Start

1. **Environment Preparation**
   - Python 3.12+
   - The project uses poetry to manage virtual environments
   - Install dependencies: `poetry install`

2. **Configure Environment**
   - Copy `.env.example` to `.env` and fill in your API key
   - Modify parameters in `models.yaml`

3. **Run System**
   ```bash
   poetry run python main.py
```

4. **Web Interface**

   ```bash
   cd web && npm install && npm run dev
   ```

   - Access http://localhost:5173 to use the web interface

## Configuration Guide

### Environment Variable Configuration

Set API keys and related configurations in the `.env` file:

```env
# Model provider API key
OPENAI_API_KEY=your_openai_api_key
# Or other provider's API key
```

### Model Configuration

System configuration file is located in `models.yaml`. Adjust the following parameters as needed:

**Optional Model Providers**

- OpenAI
- Other compatible LLM providers

**Default model and embedding model configuration used by the project**

- Default LLM model
- Default embedding model
- Model parameters (temperature, max_tokens, etc.)

**Model and embedding model configuration specifically used by project modules (optional)**

- search_agent: Search-specific model
- reading_agent: Reading-specific model
- analyse_agent: Analysis-specific model
- writing_agent: Writing-specific model
- report_agent: Report generation-specific model
- Embedding model configuration for each module

**API keys and base URLs for each model provider**

- API key configuration
- Base URL configuration
- Other connection parameters

### Configuration Example

```yaml
# models.yaml example
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

## Tech Stack

### Backend

- **Programming Language**: Python 3.12+
- **Agent Framework**:
  - AutoGen: Multi-agent collaboration framework
  - LangGraph: Workflow orchestration framework
- **Web Framework**: FastAPI, Uvicorn
- **Real-time Communication**: SSE (Server-Sent Events)
- **Vector Database**: ChromaDB
- **Data Processing**: pyyaml, python-dotenv, tenacity
- **Machine Learning**:
  - scikit-learn: KMeans clustering, elbow method
  - numpy: Vector computation
- **Paper Retrieval**: arXiv API
- **Network Requests**: requests, aiohttp
- **Package Management**: Poetry
- **Logging System**: Python standard library logging module (custom configuration)

### Frontend

- **Framework**: Vue.js 3.4+
- **Build Tool**: Vite 5.0+
- **Development Tool**: @vitejs/plugin-vue

## Contributing Guide

We welcome contributions in various forms, including but not limited to:

1. Submit issues to report bugs or suggest new features
2. Submit pull requests to improve code
3. Improve documentation

Please read [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.

## License

This project uses MIT license. See [LICENSE](../LICENSE) file for details.

## Contact

If you have any questions or suggestions, please provide feedback through:

- **GitHub Issues**: Please submit Issues in the project repository, this is the most recommended way to report issues
- Project Homepage: https://github.com/Tswoen/paper-agent

---

⭐ If this project is helpful to you, please give us a star to show your support!

## Star History

[![Star History Chart](https://api.star-history.com/image?repos=Tswoen/Paper-Agent&type=date&legend=top-left)](https://www.star-history.com/?repos=Tswoen%2FPaper-Agent&type=date&logscale=&legend=top-left)
