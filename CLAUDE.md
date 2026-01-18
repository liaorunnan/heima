# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive LLM application platform ("大模型从入门到精通") focused on:
- **RAG (Retrieval-Augmented Generation)** for English learning (vocabulary, essay writing)
- **Agent development** using LangChain with custom middleware system
- **LangGraph** workflow implementations for stateful agent orchestration
- **Multimodal AI** for image-text retrieval
- **Educational tools** including interview practice, translation, article summarization

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry (primary method)
poetry install

# Or export to requirements.txt
poetry export -f requirements.txt --output requirements.txt
pip install -r requirements.txt
```

### Running the Application
```bash
# Main FastAPI application
uvicorn main:app --host 127.0.0.1 --port 8003

# Docker build and run
docker build -t llm-rag-agent-ft .
docker run -p 8006:8006 llm-rag-agent-ft
```

### Testing & Evaluation
```bash
# RAG evaluation using RAGAS
# See rag/evaluation.py for implementation

# Run specific tutorial notebooks
jupyter notebook LangGraph_TL/1_ipynb/
jupyter notebook learn-agents-from-opencode/
```

## Architecture Overview

### RAG Pipeline
The RAG system implements a multi-stage retrieval pipeline:

```
User Query
    ↓
Query Refinement (LLM)
    ↓
Cache Check (Redis) → FAQ Check (Milvus, score > 0.91)
    ↓
Search Decision (LLM判断是否需要查询知识库)
    ↓
Parallel Retrieval:
  - Keyword Search (Elasticsearch with smartcn analyzer)
  - Vector Search (Milvus with BGE-M3 embeddings)
    ↓
Reranking
    ↓
Context Assembly
    ↓
LLM Generation (Streaming via SSE/WebSocket)
    ↓
Cache Save (Redis)
```

**Key Files:**
- `rag/rag_run_steam.py:48` - Main streaming RAG pipeline
- `rag/match_keyword.py` - Elasticsearch keyword search
- `rag/indexing.py` - Milvus vector index management
- `rag/reranker.py` - Reranking service
- `rag/prompts.py` - RAG prompt templates (4 strategies: direct, HyDE, sub-query, backtracking)

### Agent Middleware System
Custom middleware chain for agent execution:

```
User Request
    ↓
Agent (create_agent)
    ↓
Middleware Chain:
  - Dynamic Prompt Injection (memory_hooks.py)
  - Tool Retry (exponential backoff)
  - Model Retry
  - LLM Tool Emulation (mock tool calls with LLM)
  - Context Editing
  - Before/After Model Hooks
    ↓
Tool Registry
    ↓
LLM Provider (Qwen/DeepSeek/OpenAI)
    ↓
Response
```

**Key Files:**
- `langchain/toolMiddleware.py` - Middleware implementations
- `langchain/memory_hooks.py` - Memory context injection
- `langchain/LangSmith.py` - LangSmith integration

### LangGraph Workflows
State-based agent orchestration using LangGraph:
- StateGraph with TypedDict state management
- Conditional routing based on state
- Tool nodes for function calling
- Memory checkpointing with LangSmith

**Tutorials:** `LangGraph_TL/` contains learning materials (8 folders with Jupyter notebooks)

## Configuration

All configuration is centralized in `conf.py` using `pydantic-settings`:

```python
from conf import settings

# LLM Models
settings.qw_model      # Qwen model name
settings.qw_api_key     # Qwen API key
settings.qw_api_url     # Qwen API endpoint

# External Services
settings.es_host        # Elasticsearch cluster
settings.milvus_host    # Milvus vector DB
settings.redis_host     # Redis cache
settings.Emb_url        # Embedding service
settings.Rank_url       # Reranking service
```

Environment variables are loaded from `.env` file.

## API Endpoints

**Main Application (FastAPI):**
- `GET /` - Main UI
- `POST /chat` - Scenario-based chat (interview, translation, coding, business, legal)
- `POST /rag` - Non-streaming RAG query
- `POST /api/ragapi` - Streaming RAG (SSE)
- `WS /api/wsragapi` - WebSocket RAG
- `POST /jiaowuapi` - Education system RAG
- `GET /mianshi` - Random interview questions from MySQL
- `POST /zaiyao` - Article summarization
- `GET /translate` - Baidu translation API

## Key Dependencies

| Category | Libraries |
|----------|-----------|
| LLM Frameworks | langchain, langgraph, openai, dashscope |
| Vector/Search | pymilvus, elasticsearch, langchain-milvus |
| Web/API | fastapi, chainlit, uvicorn |
| ML/Data | transformers, torch, scikit-learn, ragas |
| Utilities | playwright, beautifulsoup4, redis-om, loguru |

## Development Role

This project follows the "Senior LLM Development Architect" role defined in `.trae/rules/project_rules.md`:
- Expertise in Transformer architecture, LoRA/QA fine-tuning, RLHF/DPO
- Familiarity with DeepSpeed, FSDP, vLLM, TensorRT-LLM
- Focus on RAG & Agent development with LangChain/LlamaIndex
- Emphasis on production feasibility (latency, throughput, cost)

## Code Patterns

### Streaming Responses
The project uses both SSE (Server-Sent Events) and WebSocket for streaming:

```python
# SSE endpoint (main.py:142)
StreamingResponse(
    (f"data:{json.dumps(chunk)}\n\n" async for chunk in rag_stream_run(...)),
    media_type="text/event-stream"
)

# WebSocket endpoint (main.py:148)
await websocket.send_json({"type": "token", "token": token})
```

### Parallel Retrieval
RAG uses ThreadPoolExecutor for parallel keyword and vector search (rag_run_steam.py:90):
```python
with ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(Yinyutl_rag.query, update_query)
    future2 = executor.submit(Yinyutl_Index.search, update_query)
    docs1 = future1.result()
    docs2 = future2.result()
```

### Redis Cache Pattern
QA cache using redis-om (rag_run_steam.py:54):
```python
response = QA.find(QA.query == update_query).all()
if response:
    yield from stream_answer(response[0].answer, ...)
    return
# ... save after generation
QA(query=update_query, answer=answer).save()
```

## Directory Structure Highlights

- `rag/` - Core RAG system (46 files)
- `langchain/` - Middleware and agent implementations
- `LangGraph_TL/` - LangGraph learning materials (44 files)
- `learn-agents-from-opencode/` - OpenCode agent tutorials
- `zhinengjiaowu/` - Smart education system
- `chat/` - Scenario-based chat implementations
- `chatrobot/` - Chatbot implementations
- `langragh/` - LangGraph workflows
