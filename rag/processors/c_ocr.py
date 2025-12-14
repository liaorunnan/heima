from unittest import TestCase


class Test(TestCase):
    def test_ocr(self):
        from paddleocr import PaddleOCRVL

        pipeline = PaddleOCRVL()
        output = pipeline.predict("images/12.png")
        for res in output:
            res.print()
            res.save_to_json(save_path="output")
            res.save_to_markdown(save_path="output")
            res.save_to_img(save_path="output")

    def test_chart_analyse(self):
        from paddleocr import ChartParsing

        model = ChartParsing(model_name="PP-Chart2Table")
        results = model.predict(
            input={"image": "./images/1.png"},
            batch_size=1
        )

        for res in results:
            res.save_to_json(f"./output/res.json")
            print(res)
            print(res['result'])
