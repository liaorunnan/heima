from langchain.agents import create_agent


from langchain_openai import ChatOpenAI
from conf import settings

model = ChatOpenAI(temperature=0,model_name=settings.qw_model,max_tokens=1024,timeout=60,api_key=settings.qw_api_key, base_url=settings.qw_api_url)

def send_email(to: str, subject: str, body: str):
    """Send an email"""
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # ... email sending logic

    return f"Email sent to {to}"

agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt="你是一个邮件助手，总是使用send_email工具。",
)

result = agent.invoke({"messages": [{"role": "user", "content": "请发送一封邮件给echo@example.com，主题是测试邮件，内容是这是一封测试邮件。"}]})
print(result["messages"][-1].content)
