from unittest import TestCase
from paddleocr import PaddleOCRVL

class Test(TestCase):
    def test_ocr(self):
        from paddleocr import PaddleOCRVL

        pipeline = PaddleOCRVL()
        for i in range (113):
            output = pipeline.predict(f"output/page-{i}.png")
            for res in output:
                res.print()

                res.save_to_markdown(save_path="output_markdown1")


if __name__ == '__main__':
    Test().test_ocr()