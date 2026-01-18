import os
from dataclasses import dataclass

import yaml
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from conf import settings


@dataclass
class SkillInfo:
    """Skill 信息"""
    name: str
    description: str
    location: str
    content: str


class SkillRegistry:
    """Skill 注册中心"""
    
    def __init__(self):
        self._skills: dict[str, SkillInfo] = {}
    
    def scan_skills(self, skill_dirs: list[str]) -> None:
        """扫描指定目录下的所有 SKILL.md 文件"""
        for skill_dir in skill_dirs:
            if not os.path.exists(skill_dir):
                continue
            
            # 递归查找所有 SKILL.md 文件
            for root, _, files in os.walk(skill_dir):
                if "SKILL.md" in files:
                    skill_path = os.path.join(root, "SKILL.md")
                    self._load_skill(skill_path)
    
    def _load_skill(self, skill_path: str) -> None:
        """加载单个 Skill 文件"""
        try:
            with open(skill_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析 YAML frontmatter
            if not content.startswith('---'):
                print(f"⚠️  {skill_path}: 缺少 YAML frontmatter")
                return
            
            parts = content.split('---', 2)
            if len(parts) < 3:
                print(f"⚠️  {skill_path}: frontmatter 格式错误")
                return
            
            frontmatter = yaml.safe_load(parts[1])
            skill_content = parts[2].strip()
            
            # 验证必需字段
            if 'name' not in frontmatter or 'description' not in frontmatter:
                print(f"⚠️  {skill_path}: 缺少 name 或 description 字段")
                return
            
            skill_name = frontmatter['name']
            
            # 检测重复名称
            if skill_name in self._skills:
                print(f"⚠️  重复的 Skill 名称: {skill_name}")
                print(f"   已存在: {self._skills[skill_name].location}")
                print(f"   重复项: {skill_path}")
                return
            
            # 注册 Skill
            skill_info = SkillInfo(
                name=skill_name,
                description=frontmatter['description'],
                location=skill_path,
                content=skill_content
            )
            self._skills[skill_name] = skill_info
            print(f"✅ 加载 Skill: {skill_name}")
        
        except Exception as e:
            print(f"❌ 加载 {skill_path} 失败: {e}")
    
    def get(self, name: str) -> SkillInfo | None:
        """获取指定名称的 Skill"""
        return self._skills.get(name)
    
    def all(self) -> list[SkillInfo]:
        """获取所有 Skill"""
        return list(self._skills.values())


# 全局 Skill Registry
skill_registry = SkillRegistry()


def init_skills():
    """初始化 Skill 系统"""
    print("\n=== 初始化 Skill 系统 ===")
    
    # 定义扫描目录（按优先级）
    current_dir = os.getcwd()
    skill_dirs = [
        os.path.join(current_dir, ".claude/skills"),
        os.path.expanduser("~/.claude/skills"),
        os.path.join(current_dir, ".opencode/skill"),
    ]
    
    print(f"扫描目录: {skill_dirs}")
    skill_registry.scan_skills(skill_dirs)
    print(f"共加载 {len(skill_registry.all())} 个 Skill\n")


@tool
def skill_tool(skill_name: str) -> str:
    """
    加载指定的 Skill 以获取详细指导
    
    Args:
        skill_name: Skill 标识符
    """
    print(f"\n🔧 调用 Skill Tool: {skill_name}")
    
    skill = skill_registry.get(skill_name)
    
    if not skill:
        available_skills = ", ".join([s.name for s in skill_registry.all()])
        return f"❌ Skill '{skill_name}' 未找到。可用 Skills: {available_skills or '无'}"
    
    # 打印技能名字（按需求）
    print(f"📖 加载 Skill: {skill.name}")
    
    # 格式化输出
    output = f"""
## Skill: {skill.name}

**描述**: {skill.description}
**位置**: {skill.location}

{skill.content}
"""
    return output


def create_skill_agent():
    """创建带 Skill 功能的 Agent"""
    model = ChatOpenAI(
        temperature=0.7,
        model=settings.qw_model,
        api_key=settings.qw_api_key,
        base_url=settings.qw_api_url
    )
    
    # 构建系统提示词，包含可用 Skills
    available_skills = skill_registry.all()
    skill_list = "\n".join([
        f"  - {s.name}: {s.description}"
        for s in available_skills
    ])
    
    system_prompt = f"""你是一个助手，可以使用 Skill 系统获取专业指导。

可用的 Skills:
{skill_list if skill_list else "  (暂无)"}

当用户请求需要专业知识时，使用 skill_tool 加载相应的 Skill。
"""
    
    agent = create_agent(
        model=model,
        tools=[skill_tool],
        system_prompt=system_prompt
    )
    
    return agent


def main():
    """主函数"""
    # 1. 初始化 Skill 系统
    init_skills()
    
    # 2. 列出所有可用 Skills
    print("=== 可用 Skills ===")
    for skill in skill_registry.all():
        print(f"  • {skill.name}: {skill.description}")
    print()
    
    # 3. 创建 Agent
    print("=== 创建 Agent ===")
    agent = create_skill_agent()
    print("✅ Agent 创建成功\n")
    
    # 4. 测试调用
    print("=== 测试 Skill 调用 ===")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "请帮我加载 code-reviewer skill"}]
    })
    
    print("\n=== Agent 回复 ===")
    print(result['messages'][-1].content)


if __name__ == "__main__":
    main()
