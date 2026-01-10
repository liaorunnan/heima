import requests
import numpy as np
from conf import settings

# 你的服务地址 (注意端口是 6006)
API_URL = settings.signal_emb_url 
TEST_IMAGE_PATH = "./1.png"  # 确保你本地有一张测试图片



def get_image_embedding(image_path=None,text=None):
    try:
        if image_path is None and text is None:
            raise ValueError("Either image_path or text must be provided")

        if image_path is not None:
            
            with open(image_path, "rb") as f:
                # 必须用 'file' 作为 key，对应 Server 端的 file: UploadFile
                # response = requests.post(API_URL, files={"file": f},data={"text": "图片中是什么"})
                response = requests.post(API_URL, files={"image": f})
        elif text is not None:
            response = requests.post(API_URL, data={"text": text})
       
        if response.status_code == 200:
            data = response.json()
            return np.array(data)
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"连接失败: {e}")
        return None

if __name__ == "__main__":
    print(f"正在处理图片: {TEST_IMAGE_PATH}")

    vec = get_image_embedding(TEST_IMAGE_PATH)

    # print(vec)

    vec1 = get_image_embedding(text="你好")
    print(vec1.shape)

    # print(len(vec['embedding']))
    # print(vec['dimension'])
    
    # if vec is not None:
    #     print(f"成功! 向量维度: {vec.shape}")
    #     print(f"前10位: {vec[:10]}")