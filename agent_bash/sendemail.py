import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import markdown

def send_email(title: str, email_context: str):
    # ================= 配置区域 =================
    # 1. 发件人邮箱账号
    my_sender = '774399166@qq.com'
    
    # 2. 授权码 (注意：不是邮箱登录密码！是SMTP服务的授权码)
    # 获取方式：设置 -> 账户 -> 开启POP3/SMTP服务 -> 生成授权码
    my_pass = 'zzfwvpvycahpbdeh' 
    
    # 3. 收件人邮箱账号
    my_user = '13553716073@163.com' 
    
    # 4. SMTP服务器配置 (以QQ邮箱为例)
    smtp_server = 'smtp.qq.com'
    smtp_port = 465  # SSL加密端口通常是 465
    # ===========================================

    try:
        # 创建邮件对象
        msg = MIMEMultipart()
        
        # 设置邮件头
        # formataddr((昵称, 邮箱地址)) 用来显示发件人昵称
        msg['From'] = formataddr(["热点助手", my_sender]) 
        msg['To'] = formataddr(["亲爱的用户", my_user])
        msg['Subject'] = title # 邮件主题

        # 邮件正文内容
        content = email_context
        html_content = markdown.markdown(content)
        # 将正文添加到邮件中 ('plain'代表纯文本，'html'代表html格式)
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))

        # 连接服务器并发送
        # 使用 SMTP_SSL (安全链接)
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        
        # 登录
        print("正在登录...")
        server.login(my_sender, my_pass)
        
        # 发送
        print("正在发送...")
        server.sendmail(my_sender, [my_user, ], msg.as_string())
        
        # 退出
        server.quit()
        print("邮件发送成功！")

    except Exception as e:
        print(f"邮件发送失败: {e}")

if __name__ == '__main__':
    send_email("测试邮件", "这是一封测试邮件内容")