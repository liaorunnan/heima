from unittest import TestCase


class Test(TestCase):
    def test_ocr(self):
        from paddleocr import PaddleOCRVL

        pipeline = PaddleOCRVL()
        for i in range (127):
            output = pipeline.predict(f"output/page-{i}.png")
            for res in output:
                res.print()

                res.save_to_markdown(save_path="output_markdown")


if __name__ == '__main__':
    Test().test_ocr()