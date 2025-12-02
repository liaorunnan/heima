import requests
import json


def baidu_ai_translate(query_text, from_lang="zh", to_lang="en"):
    """
    调用百度 AI 文本翻译 API。

    :param:
    query_text (str): 需要翻译的文本 (q) UTF-8编码，上限为6000字符
    app_id (str): 你的 APPID
    api_key (str): 你的 API Key (用于 Authorization Bearer)
    from_lang (str): 源语言，默认为 'zh' 可设置为auto（自动检测语言）
    to_lang (str): 目标语言，默认为 'en' 不可设置为auto

    return:
    dict: API 返回的 JSON 数据
    """

    url = "https://fanyi-api.baidu.com/ait/api/aiTextTranslate"

    app_id = "20251129002508581"
    api_key = "4SHT_d4l9ipa2l220ai5m0etg"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "appid": app_id,
        "from": from_lang,
        "to": to_lang,
        "q": query_text
    }

    try:
        response = requests.post(url, headers=headers, json=payload)


        response.raise_for_status()

        response_array =  response.json()



        return response_array['trans_result'][0]['dst']

    except requests.exceptions.RequestException as e:
        # 捕获网络异常或请求错误
        print(f"请求发生错误: {e}")
        return {"error": str(e)}


# ================= 使用示例 =================
if __name__ == "__main__":
    # 请替换为你实际的 APPID 和 API Key


    text_to_translate = "API 返回结果"

    result = baidu_ai_translate(
        query_text=text_to_translate,
        from_lang="zh",
        to_lang="en"
    )

    # 打印结果
    print("API 返回结果:")

    print(result)