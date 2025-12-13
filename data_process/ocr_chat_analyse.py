import os
from paddleocr import PaddleOCRVL
from paddleocr import ChartParsing


class ImgToMD:
    """批量图片转 Markdown/JSON 的工具类。"""
    # 初始化保存路径
    def __init__(self, output_md_dir="./output/md", output_json_dir="./output/json", output_full_md_dir="./output/full_md"):
        self.output_md_dir = output_md_dir
        self.output_json_dir = output_json_dir
        self.output_full_md_dir = output_full_md_dir
        os.makedirs(self.output_md_dir, exist_ok=True)
        os.makedirs(self.output_json_dir, exist_ok=True)
        os.makedirs(self.output_full_md_dir, exist_ok=True)
        self.model = None

    # 图片文字转 md（批量）
    def ocr_batch(self, image_paths):
        if self.model is None:
            raise RuntimeError("请先调用 load_ocr_model 加载文字识别模型")
        for image_path in image_paths:
            output = self.model.predict(image_path)
            for res in output:
                # res.print()
                basename = os.path.splitext(os.path.basename(image_path))[0]  # 提取 不带扩展名的文件名
                res.save_to_markdown(save_path=os.path.join(self.output_md_dir, f"{basename}.md"))

    # 图片图表转 md/json（批量）
    def chart_batch(self, image_paths):
        if self.model is None:
            raise RuntimeError("请先调用 load_chart_analyse 加载图表模型")
        for image_path in image_paths:
            results = self.model.predict(input={"image": image_path}, batch_size=1)
            basename = os.path.splitext(os.path.basename(image_path))[0]  # # 提取 不带扩展名的文件名
            for res in results:
                res.save_to_json(os.path.join(self.output_json_dir, f"{basename}.json"))
                res.save_to_markdown(save_path=os.path.join(self.output_md_dir, f"{basename}.md"))
                print(res)
                print(res['result'])

    def load_ocr_model(self):
        self.model = PaddleOCRVL()

    def load_chart_analyse(self):
        self.model = ChartParsing(model_name="PP-Chart2Table")

    def merge_markdown_by_pdf(self, pdf_path):
        """
        将同一 PDF 的分页 Markdown 合并为单个文件。

        假设：
        - pdf_to_images 会产出 文件名_页码.png；
        - ocr_batch 生成同名 Markdown：文件名_页码.md；
        - 本方法会在 output_md_dir 中查找以 "文件名_" 开头、以 .md 结尾的文件，按页码顺序合并到 文件名.md。
        """

        base = os.path.splitext(os.path.basename(pdf_path))[0]
        print(base)
        md_dir = self.output_md_dir

        def _page_idx(fname: str) -> int:
            stem = os.path.splitext(fname)[0]
            if "_" not in stem:
                return 0
            try:
                return int(stem.split("_")[-1])
            except ValueError:
                return 0

        md_files = [
            f for f in os.listdir(md_dir)
            if f.startswith(f"{base}_") and f.lower().endswith(".md")
        ]

        if not md_files:
            raise FileNotFoundError(f"未找到以 {base}_ 开头的 Markdown 文件，请确认 OCR 已生成分页 md")

        md_files.sort(key=_page_idx)
        merged = []
        for fname in md_files:
            with open(os.path.join(md_dir, fname), "r", encoding="utf-8") as f:
                merged.append(f.read().strip())

        output_file = os.path.join(self.output_full_md_dir, f"{base}.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(merged))

        print(f"合并完成: {output_file}，共 {len(md_files)} 页")


def list_images(folder, exts = None):
    """列出目录下的图片文件路径。"""
    if exts is None:
        exts = [".png", ".jpg", ".jpeg", ".bmp"]
    result = []
    for name in os.listdir(folder):
        if any(name.lower().endswith(e) for e in exts):
            result.append(os.path.join(folder, name))
    return result


if __name__ == '__main__':
    # 示例：批量文字 OCR
    img2md = ImgToMD()
    # img2md.load_ocr_model()
    # images = list_images("./imgs")
    # img2md.ocr_batch(images)
    img2md.merge_markdown_by_pdf("./case/1.婚姻家庭与继承纠纷.pdf")
    img2md.merge_markdown_by_pdf("./case/10.道路交通.pdf")
    img2md.merge_markdown_by_pdf("./case/11.提供劳动者受害责任纠纷.pdf")
    img2md.merge_markdown_by_pdf("./case/12.人格权纠纷.pdf")
    img2md.merge_markdown_by_pdf("./case/13.劳动纠纷.pdf")
    img2md.merge_markdown_by_pdf("./case/14.公司纠纷.pdf")
    img2md.merge_markdown_by_pdf("./case/15.保险纠纷.pdf")

    # 如需图表解析，改用下述流程：
    # img2md.load_chart_analyse()
    # img2md.chart_batch(images)