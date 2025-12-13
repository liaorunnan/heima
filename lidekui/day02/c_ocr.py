from paddleocr import PaddleOCRVL
from paddleocr import ChartParsing

class Img2text:
    def ocr(self, filepath):
        output = self.model_ocr.predict(filepath)
        for res in output:
            res.print()
            res.save_to_json(save_path="output")
            res.save_to_markdown(save_path="output")
            res.save_to_img(save_path="output")

    def chart_analyse(self, filepath):
        results = self.model_chart.predict(
            input={"image": filepath},
            batch_size=1
        )
        for res in results:
            res.save_to_json(f"./output/res.json")
            print(res)
            print(res['result'])

    def load_ocr(self):
        self.model_ocr = PaddleOCRVL()

    def load_chart(self):
        self.model_chart = ChartParsing(model_name="PP-Chart2Table")
