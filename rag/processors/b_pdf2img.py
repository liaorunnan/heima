import fitz
import os
import glob
from tqdm import tqdm


def pdf_to_images(pdf_path, image_path):
    pdfDoc = fitz.open(pdf_path)
    for pg in tqdm(range(pdfDoc.page_count)):
        page = pdfDoc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为4，这将为我们生成分辨率提高4的图像。
        zoom_x, zoom_y = 4, 4
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        if not os.path.exists(image_path):
            os.makedirs(image_path)
        pix._writeIMG(format_="png", filename=f"{image_path}/{pg}.png", jpg_quality=100)


def batch_pdf_to_images(pdf_dir, output_root):
    """
    批量将指定目录下的所有PDF文件转换为图片
    :param pdf_dir: PDF文件所在目录
    :param output_root: 图片输出根目录
    """
    # 获取目录下所有PDF文件
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    for pdf_file in tqdm(pdf_files, desc="处理PDF文件"):
        # 获取PDF文件名（不含扩展名）
        pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        # 创建对应的图片目录
        image_dir = os.path.join(output_root, pdf_name)
        
        # 检查图片目录是否已存在且包含图片文件
        if os.path.exists(image_dir):
            # 检查目录中是否有.png文件
            image_files = glob.glob(os.path.join(image_dir, "*.png"))
            if len(image_files) > 0:
                print(f"跳过已处理的PDF: {pdf_file}")
                continue
        
        # 转换PDF为图片
        print(f"正在处理: {pdf_file}")
        pdf_to_images(pdf_file, image_dir)
        print(f"处理完成，图片保存至: {image_dir}")


if __name__ == '__main__':
    # 配置参数 - 注意：PDF文件实际位于嵌套目录中
    pdf_directory = "../data/中国法院2024年度案例23册全/中国法院2024年度案例23册全"
    output_directory = "../data/image2"
    
    print(f"PDF目录: {os.path.abspath(pdf_directory)}")
    print(f"输出目录: {os.path.abspath(output_directory)}")
    
    # 执行批量转换
    batch_pdf_to_images(pdf_directory, output_directory)