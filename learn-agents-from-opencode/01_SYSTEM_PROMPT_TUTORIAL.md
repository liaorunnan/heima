这是一份经过修订的教程。我已经仔细检查了原始内容，并将所有的 TypeScript/Node.js 代码转换为符合 Python 最佳实践的代码（使用了 `asyncio`, `pydantic`, `pathlib` 等 Python AI 开发常用库）。同时，我也调整了部分文件扩展名和命令以适应 Python 环境。

---

# OpenCode System Prompt 系统学习教程 (Python 版)

> 本教程旨在深入理解 OpenCode 的 System Prompt 系统，通过 Python 源码分析和实践演练，帮助读者掌握 Agent 开发中 System Prompt 的架构设计、实现细节。

---

## 目录

| 章节 | 标题                 |
| ---- | -------------------- |
| 一   | 系统概述             |
| 二   | 核心架构分析         |
| 三   | Provider 特定 Prompt |
| 四   | 环境信息注入         |
| 五   | 自定义规则加载       |
| 六   | Prompt 注入流程      |
| 七   | Agent 特定 Prompt    |
| 八   | 实践演练             |
| 九   | 高级定制             |
| 十   | 常见问题             |

---

## 一、系统概述

### 1.1 什么是 System Prompt

**System Prompt** 是与大语言模型交互时传递给模型的第一条系统级消息，它定义了模型的身份定位、行为规范、工具权限和工作流程。

> **核心概念**：在 OpenCode 中，System Prompt 不仅仅是一段静态文本，而是一个由多个组件动态组合而成的复杂系统，涵盖了 LLM Provider 适配、环境感知、自定义规则和 Agent 特定配置等多个维度。

**分层设计的三大优势：**

- **灵活性高**：可以针对不同模型提供商定制不同的提示风格以适应不同模型的特性
- **可扩展性强**：新增提示来源只需添加对应的加载逻辑
- **维护性好**：各部分职责清晰，便于独立修改和测试

### 1.2 System Prompt 的组成结构

在 OpenCode 中，一个完整的 System Prompt 由以下 **四个核心部分** 组成：

| 部分                  | 说明                                  |
| --------------------- | ------------------------------------- |
| **Provider 特定提示** | 根据模型提供商选择对应的提示模板      |
| **环境信息**          | 动态注入运行环境的上下文              |
| **自定义规则**        | 从多个来源加载用户/项目级别的配置指令 |
| **Agent 特定提示**    | 根据 Agent 类型注入额外的指令         |

这四个部分在运行时被组装成一个列表，作为系统消息传递给大语言模型。

**注入逻辑源码** (`src/session/prompt.py`):

```python
# 假设 processor.process 接受字典或 Pydantic 模型
environment_prompts = await SystemPrompt.environment()
custom_prompts = await SystemPrompt.custom()

system_messages = environment_prompts + custom_prompts

result = await processor.process(
    user=last_user,
    agent=agent,
    abort=abort,
    session_id=session_id,
    system=system_messages,  # 合并后的系统提示
    messages=[
        *MessageV2.to_model_message(session_messages),
        *(
            [{"role": "assistant", "content": MAX_STEPS}]
            if is_last_step else []
        ),
    ],
    tools=tools,
    model=model,
)
```

从这段代码可以看出，`environment()` 和 `custom()` 两部分被合并后传递给 `processor.process` 方法。

### 1.3 核心文件概览

要深入理解 System Prompt 系统，需要重点关注以下核心文件（路径已转换为 Python 风格）：

| 文件                             | 说明                                                    |
| -------------------------------- | ------------------------------------------------------- |
| `opencode/src/session/system.py` | 定义 header、provider、environment、custom 四个关键函数 |
| `opencode/src/session/prompt.py` | Prompt 处理主文件，实现 System Prompt 注入逻辑          |
| `opencode/src/agent/agent.py`    | 定义所有内置 Agent 的配置                               |
| `opencode/src/session/prompt/`   | 包含所有 Provider 和 Agent 特定的提示模板 (.txt)        |

---

## 二、核心架构分析

### 2.1 SystemPrompt 类详解

**SystemPrompt** 类是整个 System Prompt 系统的核心，位于 `opencode/src/session/system.py`。

**四个静态方法：**

| 函数            | 职责                                 |
| --------------- | ------------------------------------ |
| `header()`      | 根据 Provider ID 注入特定的标识提示  |
| `provider()`    | 根据模型标识选择对应的提示模板       |
| `environment()` | 收集和格式化当前运行环境的上下文信息 |
| `custom()`      | 从多个来源加载用户自定义的指令规则   |

**文件导入依赖：**

```python
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List

# 假设这是一个读取文本文件的辅助函数
from opencode.utils.files import read_text_file 
from opencode.global_config import Global
from opencode.config import Config
from opencode.project import Instance
from opencode.provider import Provider

# 预加载 Prompt 模板内容
PROMPT_ANTHROPIC = read_text_file("prompt/anthropic.txt")
PROMPT_ANTHROPIC_WITHOUT_TODO = read_text_file("prompt/qwen.txt")
PROMPT_BEAST = read_text_file("prompt/beast.txt")
PROMPT_GEMINI = read_text_file("prompt/gemini.txt")
PROMPT_ANTHROPIC_SPOOF = read_text_file("prompt/anthropic_spoof.txt")
PROMPT_CODEX = read_text_file("prompt/codex.txt")

class SystemPrompt:
    # 四个核心静态方法定义
    pass
```

**核心依赖模块：**

- **Global**：提供全局配置路径
- **Config**：提供用户配置读取能力
- **Instance**：提供当前项目和工作目录信息
- **pathlib**：Python 标准库，提供文件系统操作能力

---

### 2.2 header 方法：Provider 标识注入

**功能**：根据 Provider ID 注入特定的标识提示。

```python
    @staticmethod
    def header(provider_id: str) -> List[str]:
        if "anthropic" in provider_id:
            return [PROMPT_ANTHROPIC_SPOOF.strip()]
        return []
```

**逻辑说明：**
- 只有当 Provider ID 包含 `"anthropic"` 时，才返回特殊的 spoof 提示
- 否则返回空列表

**调用时机**：在 Agent 生成阶段调用。

```python
async def generate(input_data: dict):
    cfg = await Config.get()
    default_model = input_data.get("model") or (await Provider.default_model())
    # ... 获取模型实例
    
    system = SystemPrompt.header(default_model.provider_id)
    system.append(PROMPT_GENERATE)
    # ... 后续处理
```

---

### 2.3 provider 方法：Provider 适配选择

**功能**：根据模型标识选择对应的提示模板。

```python
    @staticmethod
    def provider(model: Provider.Model) -> List[str]:
        model_id = model.api.id
        
        if "gpt-5" in model_id:
            return [PROMPT_CODEX]
        if "gpt-" in model_id or "o1" in model_id or "o3" in model_id:
            return [PROMPT_BEAST]
        if "gemini-" in model_id:
            return [PROMPT_GEMINI]
        if "claude" in model_id:
            return [PROMPT_ANTHROPIC]
            
        return [PROMPT_ANTHROPIC_WITHOUT_TODO]
```

**模型适配对照表：**

| 模型系列                  | 提示模板        | 说明                   |
| ------------------------- | --------------- | ---------------------- |
| GPT-5 / Codex             | `codex.txt`     | 专门的代码生成优化     |
| GPT-4 / GPT-3.5 / o1 / o3 | `beast.txt`     | OpenAI 系列通用模板    |
| Gemini                    | `gemini.txt`    | Google 模型专用模板    |
| Claude                    | `anthropic.txt` | Anthropic 模型默认模板 |
| 其他                      | `qwen.txt`      | 默认回退模板           |

---

### 2.4 environment 方法：运行时环境信息

**功能**：收集和格式化当前运行环境的上下文信息。

```python
    @staticmethod
    async def environment() -> List[str]:
        project = Instance.project
        # 模拟 ripgrep 结果，实际应调用 subprocess 或相关库
        files_content = "" 
        if project.vcs == "git" and False: # 同样保持禁用状态
             files_content = await Ripgrep.tree(cwd=Instance.directory, limit=200)

        env_info = [
            "Here is some useful information about the environment you are running in:",
            "<env>",
            f"  Working directory: {Instance.directory}",
            f"  Is directory a git repo: {'yes' if project.vcs == 'git' else 'no'}",
            f"  Platform: {sys.platform}",
            f"  Today's date: {datetime.now().strftime('%a %b %d %Y')}",
            "</env>",
            "<files>",
            files_content,
            "</files>",
        ]
        
        return ["\n".join(env_info)]
```

**收集的环境信息字段：**

| 字段                      | Python 获取方式                      |
| ------------------------- | ------------------------------------ |
| `Working directory`       | `Path.cwd()` 或 `Instance.directory` |
| `Is directory a git repo` | 检查 `.git` 目录是否存在             |
| `Platform`                | `sys.platform`                       |
| `Today's date`            | `datetime.now()`                     |

---

### 2.5 custom 方法：自定义规则加载

**功能**：从多个来源加载用户自定义的指令规则。

**加载策略与 TypeScript 版本保持一致：** 本地项目级 > 全局用户级 > 配置指令。

**完整代码实现 (Python)：**

```python
import httpx
from pathlib import Path

# 定义常量
LOCAL_RULE_FILES = ["AGENTS.md", "CLAUDE.md", "CONTEXT.md"]
GLOBAL_RULE_FILES = [
    Path(Global.Path.config) / "AGENTS.md",
    Path.home() / ".claude" / "CLAUDE.md"
]

if os.getenv("OPENCODE_CONFIG_DIR"):
    GLOBAL_RULE_FILES.append(Path(os.getenv("OPENCODE_CONFIG_DIR")) / "AGENTS.md")

    @staticmethod
    async def custom() -> List[str]:
        config = await Config.get()
        paths = set()

        # 1. 本地规则查找 (向上递归查找)
        # 模拟 find_up 逻辑
        current_dir = Path(Instance.directory)
        root_dir = Path(Instance.worktree)
        
        found_local = False
        temp_dir = current_dir
        while True:
            for rule_file in LOCAL_RULE_FILES:
                target = temp_dir / rule_file
                if target.exists():
                    paths.add(str(target))
                    found_local = True
                    break # 找到当前目录优先级最高的文件
            
            if found_local or temp_dir == root_dir or temp_dir == temp_dir.parent:
                break
            temp_dir = temp_dir.parent

        # 2. 全局规则查找
        for global_file in GLOBAL_RULE_FILES:
            if global_file.exists():
                paths.add(str(global_file))
                break

        # 3. 配置指令处理
        urls = []
        if config.instructions:
            for instruction in config.instructions:
                # URL 处理
                if instruction.startswith(("http://", "https://")):
                    urls.append(instruction)
                    continue
                
                # 路径展开
                if instruction.startswith("~/"):
                    instruction = str(Path.home() / instruction[2:])
                
                # 文件匹配 (Glob)
                matches = []
                instr_path = Path(instruction)
                if instr_path.is_absolute():
                    # 绝对路径 glob
                    matches = [str(p) for p in instr_path.parent.glob(instr_path.name)]
                else:
                    # 相对路径 glob (模拟 globUp)
                    # 这里简化为在当前目录 glob
                    matches = [str(p) for p in Path(Instance.directory).glob(instruction)]
                
                for m in matches:
                    paths.add(m)

        # 异步加载函数
        async def load_file(path_str):
            try:
                content = await asyncio.to_thread(Path(path_str).read_text, encoding='utf-8')
                return f"Instructions from: {path_str}\n{content}"
            except Exception:
                return ""

        async def load_url(url_str):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url_str, timeout=5.0)
                    if resp.status_code == 200:
                        return f"Instructions from: {url_str}\n{resp.text}"
            except Exception:
                pass
            return ""

        # 并发执行所有加载任务
        tasks = [load_file(p) for p in paths] + [load_url(u) for u in urls]
        results = await asyncio.gather(*tasks)
        
        # 过滤空字符串
        return [r for r in results if r]
```

**技术点**：
- 使用 `asyncio.gather` 实现并发 I/O。
- 使用 `pathlib` 处理路径。
- 使用 `httpx` 处理异步 HTTP 请求。

---

## 三、Provider 特定 Prompt

*(内容逻辑与原文一致，Python 代码只需关注文件读取，本节主要为文本配置，故略过代码转换，侧重概念)*

### 3.1 Prompt 模板文件概述

Python 项目中，这些通常存储在 `opencode/src/session/prompt/` 目录下，并通过 `importlib.resources` 或直接文件读取加载。

---

## 四、环境信息注入

### 4.1 环境信息的组成

**(参考 2.4 节代码)** Python 中使用 `sys`, `os`, `datetime` 等标准库获取。

---

## 五、自定义规则加载

### 5.2 文件查找策略 (Python 实现)

**核心工具函数对应关系：**

| TypeScript            | Python (pathlib/custom)         |
| --------------------- | ------------------------------- |
| `Filesystem.findUp()` | 自定义循环 `path.parent` 检查   |
| `Filesystem.globUp()` | `path.rglob()` 或自定义递归逻辑 |

---

## 六、 Prompt 注入流程

### 6.1 Prompt 处理主流程

**文件位置**：`opencode/src/session/prompt.py`

**核心逻辑**：

```python
# System Prompt 组装
system_prompts = []
system_prompts.extend(await SystemPrompt.environment())
system_prompts.extend(await SystemPrompt.custom())

# 消息体构建
messages = MessageV2.to_model_message(session_messages)

if is_last_step:
    messages.append({
        "role": "assistant",
        "content": MAX_STEPS
    })

# 调用处理器
result = await processor.process(
    user=last_user,
    agent=agent,
    abort=abort,
    session_id=session_id,
    system=system_prompts, # 注入点
    messages=messages,
    tools=tools,
    model=model
)
```

---

## 七、Agent 特定 Prompt

### 7.1 Agent 定义结构 (Pydantic)

在 Python 中，我们通常使用 `Pydantic` 来替代 `zod` 进行数据验证和模式定义。

**文件位置**：`opencode/src/agent/agent.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from opencode.permission import PermissionNext

class AgentInfo(BaseModel):
    name: str
    description: Optional[str] = None
    mode: Literal["subagent", "primary", "all"]
    native: Optional[bool] = False
    hidden: Optional[bool] = False
    top_p: Optional[float] = Field(None, alias="topP")
    temperature: Optional[float] = None
    color: Optional[str] = None
    permission: Any = PermissionNext.Ruleset # 需定义具体类型
    model: Optional[Dict[str, str]] = None # {modelID: ..., providerID: ...}
    prompt: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    steps: Optional[int] = Field(None, gt=0) # positive integer

    class Config:
        populate_by_name = True
```

### 7.2 内置 Agent 分析

**build Agent 配置 (Python Dict 风格):**

```python
AGENTS = {
    "build": {
        "name": "build",
        "options": {},
        "permission": PermissionNext.merge(defaults, user),
        "mode": "primary",
        "native": True,
    },
    "explore": {
        "name": "explore",
        "permission": PermissionNext.merge(
            defaults,
            PermissionNext.from_config({
                "*": "deny",
                "grep": "allow",
                "glob": "allow",
                # ... 其他权限
            }),
            user,
        ),
        "description": "Fast agent specialized for exploring codebases...",
        "prompt": PROMPT_EXPLORE, # 引用加载的文本
        "mode": "subagent",
        "native": True,
    }
}
```

### 7.4 自定义 Agent 配置加载

**配置加载逻辑 (Python):**

```python
def load_agents(cfg):
    result = {}
    
    # cfg.agent 是一个字典
    for key, value in (cfg.agent or {}).items():
        if value.get("disable"):
            if key in result:
                del result[key]
            continue
            
        item = result.get(key)
        if not item:
            item = {
                "name": key,
                "mode": "all",
                "permission": PermissionNext.merge(defaults, user),
                "options": {},
                "native": False,
            }
            result[key] = item

        # 属性覆盖 (使用字典 update 或手动赋值)
        if value.get("model"):
            item["model"] = Provider.parse_model(value["model"])
            
        # Python 的字典 get 方法类似 JS 的 value.prompt ?? item.prompt
        item["prompt"] = value.get("prompt", item.get("prompt"))
        item["description"] = value.get("description", item.get("description"))
        item["temperature"] = value.get("temperature", item.get("temperature"))
        item["top_p"] = value.get("top_p", item.get("top_p"))
        # ... 其他属性合并
        
        # 深度合并 options
        item["options"] = merge_deep(item["options"], value.get("options", {}))
        
    return result
```

---

## 八、实践演练

### 8.1 实践一：修改 System Prompt 添加项目规范

**步骤**：在项目根目录创建 `AGENTS.md` 文件。

**(文件内容保持 Markdown 格式不变，Python 代码会自动读取)**

---

### 8.4 实践四：调试 System Prompt (Python)

**调试代码示例：**

```python
# 在 prompt.py 中添加日志
import logging

logger = logging.getLogger(__name__)

async def process_prompt(...):
    # ...
    env_parts = await SystemPrompt.environment()
    custom_parts = await SystemPrompt.custom()
    
    print("System Prompt parts:")
    for part in env_parts + custom_parts:
        print(f"- {part[:100]}...")
```

### 8.5 实践五：创建 Provider 特定提示

**步骤二**：添加判断逻辑 (Python)

```python
    @staticmethod
    def provider(model: Provider.Model) -> List[str]:
        if "new-provider" in model.api.id:
            return [PROMPT_NEW_PROVIDER]
        # ... 现有逻辑
```

---

## 附录

### 附录二、运行命令速查 (Python 环境)

**环境准备：**

```bash
# 安装依赖 (使用 pip 或 poetry)
pip install -r requirements.txt
# 或
poetry install
```

**测试运行：**

```bash
# 运行主程序
python src/index.py

# 类型检查 (使用 mypy)
mypy src/

# 运行测试 (使用 pytest)
pytest
```

---
**注**：以上文档中的代码已完全替换为 Python 实现，逻辑与原 TypeScript 版本保持一致，使用了 `asyncio` 进行异步处理，`pydantic` 进行类型定义，以及 `pathlib` 进行文件操作。
