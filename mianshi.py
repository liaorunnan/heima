import pymysql
import random

# 1. 数据库配置 (保持不变)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '08aba293f13b4bab',
    'database': 'heima',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 2. 数据库执行函数 (保持不变)
def execute_sql_safe(sql, params=None):
    result = {'success': False, 'data': None, 'error': None}
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        if sql.strip().upper().startswith('SELECT'):
            result['data'] = cursor.fetchall()
        else:
            conn.commit()
            result['data'] = cursor.rowcount
            
        result['success'] = True

    except Exception as e:
        result['error'] = str(e)
        print(f"Database Error: {e}")
    finally:
        if conn:
            conn.close()
            
    return result


def get_random_question_logic(types_str):
    interview_questions = None


    
    if types_str:
        type_list = types_str.split(',')
        type_list = [t.strip() for t in type_list if t.strip()]
        
        if type_list:
            placeholders = ', '.join(['%s'] * len(type_list))
            sql = f"SELECT * FROM mianshi WHERE active = 1 AND type IN ({placeholders})"
            params = type_list
        else:
            sql = "SELECT * FROM mianshi WHERE active = 1"
            params = None
    else:
        sql = "SELECT * FROM mianshi WHERE active = 1"
        params = None

    # 执行查询
    result = execute_sql_safe(sql, params)

    if result['success']:
        interview_questions = result['data']

    # 处理无数据的情况
    if not interview_questions:
    
        return {"question": "1111", "answer": "", "type": "error", "error": sql}

    # 随机选择
    random_question = random.choice(interview_questions)

    # 返回纯数据字典
    return {
        "question": random_question.get('question'),
        "answer": random_question.get('answer'),
        "type": random_question.get('type'),
    }

