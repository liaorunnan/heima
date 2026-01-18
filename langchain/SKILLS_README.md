# LangChain Skill 系统 Demo

## 📖 概述

这是一个基于 OpenCode Skill 系统设计理念的简化版实现，使用 LangChain 框架实现。

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Skill 系统工作流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 扫描 SKILL.md 文件                                       │
│     └── .claude/skills/**/SKILL.md                         │
│     └── ~/.claude/skills/**/SKILL.md                       │
│     └── .opencode/skill/**/SKILL.md                        │
│                                                             │
│  2. 解析 YAML Frontmatter                                   │
│     └── name: skill-identifier                             │
│     └── description: Skill 描述                             │
│                                                             │
│  3. 注册到 SkillRegistry                                    │
│     └── skills: dict[name -> SkillInfo]                    │
│                                                             │
│  4. 暴露为 LangChain Tool                                   │
│     └── skill_tool(skill_name: str) -> str                 │
│                                                             │
│  5. Agent 调用 Skill                                        │
│     └── 打印技能名字 + 返回技能内容                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 目录结构

```
app/
├── langchain/
│   ├── skills.py              # Skill 系统核心实现
│   └── SKILLS_README.md       # 本文档
│
└── .claude/
    └── skills/
        ├── code-reviewer/
        │   └── SKILL.md       # 代码审查 Skill
        └── python-expert/
            └── SKILL.md       # Python 专家 Skill
```

## 🔑 核心组件

### 1. SkillInfo (数据类)
```python
@dataclass
class SkillInfo:
    name: str          # Skill 唯一标识符
    description: str   # Skill 功能描述
    location: str      # SKILL.md 文件路径
    content: str       # Skill 详细内容
```

### 2. SkillRegistry (注册中心)
负责扫描、加载和管理所有 Skill：

- `scan_skills(skill_dirs)`: 扫描指定目录
- `get(name)`: 获取单个 Skill
- `all()`: 获取所有 Skill

### 3. skill_tool (LangChain Tool)
暴露给 Agent 的工具函数：

```python
@tool
def skill_tool(skill_name: str) -> str:
    """加载指定的 Skill 以获取详细指导"""
    # 1. 打印技能名字
    # 2. 返回格式化的技能内容
```

### 4. create_skill_agent (Agent 创建)
创建带 Skill 功能的 LangChain Agent

## 📝 SKILL.md 格式

每个 Skill 必须遵循以下格式：

```markdown
---
name: skill-identifier
description: Skill 功能描述
---

# Skill 标题

## 何时使用
- 使用场景 1
- 使用场景 2

## 使用步骤
1. 步骤一
2. 步骤二

## 最佳实践
- 实践建议
```

**必需字段：**
- `name`: Skill 唯一标识符（kebab-case 风格）
- `description`: 简短的功能描述，用于 Agent 选择 Skill

## 🚀 使用方式

### 运行 Demo

```bash
cd /Users/echo/Documents/cyb/test/dockertest/app/langchain
python skills.py
```

### 预期输出

```
=== 初始化 Skill 系统 ===
扫描目录: [...]
✅ 加载 Skill: code-reviewer
✅ 加载 Skill: python-expert
共加载 2 个 Skill

=== 可用 Skills ===
  • code-reviewer: 代码审查专家，用于检测 bug、安全漏洞和代码风格问题
  • python-expert: Python 开发专家，提供性能优化、最佳实践和库选择建议

=== 创建 Agent ===
✅ Agent 创建成功

=== 测试 Skill 调用 ===
🔧 调用 Skill Tool: code-reviewer
📖 加载 Skill: code-reviewer

=== Agent 回复 ===
[包含 Skill 详细内容]
```

## 🎯 关键实现细节

### 1. 扫描流程
```python
for root, _, files in os.walk(skill_dir):
    if "SKILL.md" in files:
        skill_path = os.path.join(root, "SKILL.md")
        self._load_skill(skill_path)
```

### 2. YAML 解析
```python
parts = content.split('---', 2)
frontmatter = yaml.safe_load(parts[1])
skill_content = parts[2].strip()
```

### 3. 重名检测
```python
if skill_name in self._skills:
    print(f"⚠️  重复的 Skill 名称: {skill_name}")
    return  # 跳过重复项
```

### 4. 工具集成
```python
@tool
def skill_tool(skill_name: str) -> str:
    print(f"🔧 调用 Skill Tool: {skill_name}")
    skill = skill_registry.get(skill_name)
    # 格式化并返回内容
```

## 🔍 与 OpenCode 的对应关系

| OpenCode | LangChain Demo |
|----------|---------------|
| `Skill.state()` | `SkillRegistry.__init__()` |
| `Skill.get()` | `SkillRegistry.get()` |
| `Skill.all()` | `SkillRegistry.all()` |
| `SkillTool` | `skill_tool` |
| `ConfigMarkdown.parse()` | YAML + 字符串分割 |
| 权限过滤 | 未实现（简化） |

## 🛠️ 自定义 Skill

### 创建新 Skill

```bash
# 1. 创建目录
mkdir -p .claude/skills/my-skill

# 2. 创建 SKILL.md
cat > .claude/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: 我的自定义 Skill
---

# 我的 Skill

## 使用说明
...
EOF

# 3. 重新运行程序
python langchain/skills.py
```

## ⚠️ 注意事项

1. **文件编码**: SKILL.md 必须是 UTF-8 编码
2. **YAML 语法**: frontmatter 必须是有效的 YAML
3. **名称唯一**: 相同名称的 Skill 后加载的会被跳过
4. **目录扫描**: 仅扫描指定的 3 个目录

## 📚 扩展建议

如需扩展此 Demo，可考虑：

- ✅ 添加权限控制（参考 OpenCode 的 Permission 系统）
- ✅ 支持 Skill 参数化
- ✅ 添加 Skill 热加载
- ✅ 实现 Skill 版本管理
- ✅ 支持 Skill 依赖关系

## 🎓 学习资源

- OpenCode Skill 教程: `learn-agents-from-opencode/06_SKILL_SYSTEM_TUTORIAL.md`
- LangChain Tools 文档: https://python.langchain.com/docs/how_to/custom_tools/
