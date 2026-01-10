from PIL import Image
grid_x,grid_y = 2,2
def multimodal_photo_cut(image_path, target_size=336):
    # 打开图片
    image = Image.open(image_path)

    # 获取图片的宽度和高度
    width, height = image.size

    # 缩放图片
    gloal_img = image.resize(size=(target_size,target_size))

    #切成几分
    grid_x,grid_y = 2,2
    step_x,step_y = width//2,height//2

    # 图片集
    patches = []

    # 切图 先y后x，符合视觉习惯，一行看完再看下一行
    for i in range(grid_y):
        for j in range(grid_x):
            # 拿到图片的四个点
            box = (i*step_x, j*step_y, (i+1)*step_x, (j+1)*step_y)
            # 裁剪图片
            patch = image.crop(box).resize(size=(target_size,target_size))
            # 加入图片集
            patches.append(patch)

    return gloal_img,patches

def photo_emb(path_dir):
    pass

if __name__ == '__main__':
    gloal_img,patches = multimodal_photo_cut('./Multimodal_photo.png')

    num_array = [(i,j) for i in range(grid_y) for j in range(grid_x)]
    
    for i,patch in enumerate(patches):
        print(patch)
        patch.save(f'./patch_{i}.png')
       

    # gloal_img.show()