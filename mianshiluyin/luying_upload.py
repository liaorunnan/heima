# -*- coding=utf-8 -*-
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
import datetime

# ================= 配置区域 =================
# 1. 替换为你的 SecretId 和 SecretKey
SECRET_ID = '你的SecretId'      
SECRET_KEY = '你的SecretKey'    

# 2. 替换为你的存储桶地域（例如广州是 ap-guangzhou，北京是 ap-beijing）
REGION = 'ap-guangzhou' 

# 3. 替换为你的 Bucket 名称（格式：名称-APPID）
BUCKET_NAME = 'temp-audio-1250000000' 

# ================= 初始化客户端 =================
# Token 参数传 None (如果使用临时密钥需要传 Token，长期密钥传 None)
config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY, Token=None)
client = CosS3Client(config)

def upload_temp_audio(local_file_path):
    """
    上传音频并获取临时链接
    :param local_file_path: 本地录音文件路径
    :return: 临时访问 URL
    """
    # 1. 生成云端文件名（建议带时间戳，防止重名）
    # 存放在 audio/ 目录下
    file_name = f"audio/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{os.path.basename(local_file_path)}"

    try:
        print(f"正在上传: {local_file_path} ...")
        
        # 2. 上传文件
        # EnableMD5=False 可以略微提升速度，大文件建议开启
        response = client.upload_file(
            Bucket=BUCKET_NAME,
            LocalFilePath=local_file_path,
            Key=file_name,
            EnableMD5=False
        )
        
        # 3. 生成临时预签名 URL (Presigned URL)
        # 就算 Bucket 是私有的，拥有这个链接的人在 expired 时间内也能下载/播放
        # expired=1800 表示链接 1800秒 (30分钟) 后失效
        url = client.get_presigned_url(
            Method='GET',
            Bucket=BUCKET_NAME,
            Key=file_name,
            Expired=1800 
        )
        
        print("上传成功！")
        return url

    except Exception as e:
        print(f"上传失败: {e}")
        return None

# ================= 测试运行 =================
if __name__ == '__main__':
    # 假设你本地有一个录音文件叫 test_record.wav
    # 请确保该文件存在，或者换成你真实的路径
    local_audio = "test_record.wav" 
    
    # 为了测试，先创建一个空文件
    if not os.path.exists(local_audio):
        with open(local_audio, "wb") as f:
            f.write(b"fake audio content")

    download_url = upload_temp_audio(local_audio)
    
    if download_url:
        print("-" * 30)
        print("给前端或API使用的临时链接 (30分钟有效):")
        print(download_url)
        print("-" * 30)
        print("注意：根据生命周期规则，该文件将在 1 天后从服务器物理删除。")