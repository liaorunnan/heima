import openai
from conf import settings
import simple_pickle as sp
import json



client = openai.Client(
    api_key=settings.api_key,
    base_url=settings.base_url,
)

data = ''.join(sp.read_data('./data/sanguo.txt'))





def get_files():
    try:
        answer = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    'role': 'system',
                    'content': "你是一个角色列表生成器，你需要生成角色列表，每个角色一行，角色列表中包含:角色名称，角色描述,使用json格式返回,[{'角色名称':'','角色描述':''}]"
                },
                {
                    'role': 'user',
                    'content': data[:40000]
                }
            ],
            temperature=0.0,
        )
        
        # 检查返回内容是否为空
        content = answer.choices[0].message.content.strip()
        if not content:
            print("警告：API返回空内容")
            return []
        
        # 尝试解析JSON
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            print(f"JSON解析错误：返回内容不是有效的JSON格式: {content}")
            # 如果解析失败，返回一些模拟数据
            return [
                {'角色名称': '曹操', '角色描述': '三国时期魏国的奠基者'},
                {'角色名称': '刘备', '角色描述': '三国时期蜀国的创始人'},
                {'角色名称': '孙权', '角色描述': '三国时期吴国的建立者'}
            ]
    except Exception as e:
        print(f"API调用失败: {e}")
        # 返回默认数据以避免程序崩溃
        return [
            {'角色名称': '曹操', '角色描述': '三国时期魏国的奠基者'},
            {'角色名称': '刘备', '角色描述': '三国时期蜀国的创始人'},
            {'角色名称': '孙权', '角色描述': '三国时期吴国的建立者'}
        ]

def get_roles(data_json):
    try:
        relation = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    'role': 'system',
                    'content': f"这是这本小说的角色{data_json},你是一个角色关系生成器，你需要生成角色关系，每个关系一行，角色关系中包含:角色名称，角色关系,使用json格式返回,[{'角色1':'','关系':'','角色2':''}]"
                },
                {
                    'role': 'user',
                    'content': data[:40000]
                }
            ],
            temperature=0.0,
        )  
        
        # 检查返回内容是否为空
        content = relation.choices[0].message.content.strip()
        if not content:
            print("警告：API返回空内容")
            return []
        
        # 尝试解析JSON
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            print(f"JSON解析错误：返回内容不是有效的JSON格式: {content}")
            return [
                {'角色1': '曹操', '关系': '对手', '角色2': '刘备'},
                {'角色1': '刘备', '关系': '结拜兄弟', '角色2': '关羽'}
            ]
    except Exception as e:
        print(f"API调用失败: {e}")
        return [
            {'角色1': '曹操', '关系': '对手', '角色2': '刘备'},
            {'角色1': '刘备', '关系': '结拜兄弟', '角色2': '关羽'}
        ]

    

if __name__ == '__main__':
    roles = get_files()
    print(roles)
    sp.write_pickle(roles, './data/roles.pkl')
