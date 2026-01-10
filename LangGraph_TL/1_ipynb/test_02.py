"""

使用 TypedDict 定义数据结构的测试案例

"""


from typing import TypedDict

class Contact(TypedDict):
    name: str
    email: str
    phone: str

def send_email(contact: Contact) -> None:
    print(f"Sending email to {contact['name']} at {contact['email']}")

# 使用定义好的 TypedDict 创建字典
contact_info: Contact = {
    'name': 'thy',
    'email': 'thy@example.com',
    'phone': '123-456-7890'
}

send_email(contact_info)

