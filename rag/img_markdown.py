from paddleocr import PaddleOCRVL
import
pipeline = PaddleOCRVL()

for i in range(1,21):
    output = pipeline.predict(f"data/english_wenzhang/{i}.jpg")
    print(output)
    for res in output:
        res.print() ## 打印预测的结构化输出
        res.save_to_json(save_path="wenzhang") ## 保存当前图像的结构化json结果
        res.save_to_markdown(save_path="wenzhang") ## 保存当前图像的markdown格式的结果