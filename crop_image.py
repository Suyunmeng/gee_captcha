from PIL import Image, ImageFont, ImageDraw, ImageOps
from io import BytesIO
import numpy as np
import os
current_path = os.getcwd()
validate_path = os.path.join(current_path,'img_2_val')#要验证的图片暂存
save_path = os.path.join(current_path,'img_saved')#存放历史图片，留作做数据集以待标记
save_pass_path = os.path.join(save_path,'img_pass')#校验失败的图片，可能是轨迹有误，不一定是分类错误
save_fail_path = os.path.join(save_path,'img_fail')#校验成功的图片，但有可能有个别分类错误
os.makedirs(validate_path,exist_ok=True)
os.makedirs(save_path,exist_ok=True)
os.makedirs(save_pass_path,exist_ok=True)
os.makedirs(save_fail_path,exist_ok=True)
def draw_points_on_image(bg_image, answer):
    import cv2
    # 将背景图片转换为OpenCV格式
    bg_image_cv = cv2.imdecode(np.frombuffer(bg_image, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # 定义九宫格的大小和偏移量
    grid_width = 100
    grid_height = 86
    offset_x = 45
    offset_y = 38
    
    for i, (row, col) in enumerate(answer):
        x = offset_x + (col-1) * grid_width
        y = offset_y + (row-1) * grid_height
        cv2.circle(bg_image_cv, (x, y), 10, (0, 0, 255), -1)
    cv2.imwrite('./img_2_val/predicted.jpg', bg_image_cv)#推理结果

def convert_png_to_jpg(png_bytes: bytes) -> bytes:
    # 将传入的 bytes 转换为图像对象
    png_image = Image.open(BytesIO(png_bytes))

    # 创建一个 BytesIO 对象，用于存储输出的 JPG 数据
    output_bytes = BytesIO()

    # 检查图像是否具有透明度通道 (RGBA)
    if png_image.mode == 'RGBA':
        # 创建白色背景
        white_bg = Image.new("RGB", png_image.size, (255, 255, 255))
        # 将 PNG 图像粘贴到白色背景上，透明部分用白色填充
        white_bg.paste(png_image, (0, 0), png_image)
        jpg_image = white_bg
    else:
        # 如果图像没有透明度，直接转换为 RGB 模式
        jpg_image = png_image.convert("RGB")

    # 将转换后的图像保存为 JPG 格式到 BytesIO 对象
    jpg_image.save(output_bytes, format="JPEG")

    # 返回保存后的 JPG 图像的 bytes
    return output_bytes.getvalue()

def bytes_to_pil(image_bytes):
    image  = Image.open(BytesIO(image_bytes))
    image = image.convert('RGB')
    return image

def crop_image(image_bytes, coordinates):
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    grid_width = width // 3
    grid_height = height // 3
    cropped_images = []
    for coord in coordinates:
        y, x = coord
        left = (x - 1) * grid_width
        upper = (y - 1) * grid_height
        right = left + grid_width
        lower = upper + grid_height
        box = (left, upper, right, lower)
        cropped_img = img.crop(box)
        cropped_images.append(cropped_img)
    return cropped_images

def crop_image_v3(image_bytes):
    coordinates = [
                    [[0, 0], [112, 112]],
                    [[116, 0], [228, 112]],
                    [[232, 0], [344, 112]],#第一行
                    [[0, 116], [112, 228]],
                    [[116, 116], [228, 228]],
                    [[232, 116], [344, 228]],#第二行
                    [[0, 232], [112, 344]],
                    [[116, 232], [228, 344]],
                    [[232, 232], [344, 344]],#第三行
                    [[2, 344], [42, 384]] #要验证的
                  ]
    image  = bytes_to_pil(image_bytes)
    imageNew = Image.new('RGB', (300,261),(0,0,0))
    images = []
    for i, (start_point, end_point) in enumerate(coordinates):
        x1, y1 = start_point
        x2, y2 = end_point
        # 切割图像
        cropped_image = image.crop((x1, y1, x2, y2))
        images.append(cropped_image)
        # 保存切割后的图像
        output_path = os.path.join(validate_path,f'cropped_{i}.jpg') 
        cropped_image.save(output_path)
    for i in range(3):
        imageNew.paste(images[i].resize((100,86)), (i*100, 0, (i+1)*100, 86))
        imageNew.paste(images[i+3].resize((100,86)), (i*100, 86, (i+1)*100, 172))
        imageNew.paste(images[i+6].resize((100,86)), (i*100,172, (i+1)*100, 258))
    imageNew.save(os.path.join(validate_path,f'nine.jpg') )
if __name__ == "__main__":
    # v4测试代码
    # os.makedirs(os.path.join(current_path,'image_test'),exist_ok=True)
    # # 切割顺序，这里是从左到右，从上到下[x,y]
    # coordinates = [
    #     [1, 1],
    #     [1, 2],
    #     [1, 3],
    #     [2, 1],
    #     [2, 2],
    #     [2, 3],
    #     [3, 1],
    #     [3, 2],
    #     [3, 3],
    # ]
    # with open("./image_test/bg.jpg", "rb") as rb:
    #     bg_img = rb.read()
    # cropped_images = crop_image(bg_img, coordinates)
    # # 一个个保存下来
    # for j, img_crop in enumerate(cropped_images):
    #     img_crop.save(f"./image_test/bg{j}.jpg")
    
    # # 图标格式转换
    # with open("./image_test/icon.png", "rb") as rb:
    #     icon_img = rb.read()
    # icon_img_jpg = convert_png_to_jpg(icon_img)
    # with open("./image_test/icon.jpg", "wb") as wb:
    #     wb.write(icon_img_jpg)
    
    # V3测试代码
    pic = "img_saved/7fe559a85bac4c03bc6ea7b2e85325bf.jpg"
    print("推理图片为：",pic)
    with open(pic, "rb") as f:
        img = f.read()
        crop_image_v3(img)