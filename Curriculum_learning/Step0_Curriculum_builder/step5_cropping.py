import os
from PIL import Image

def crop_image(image_path, output_path, crop_params):
    with Image.open(image_path) as img:
        width, height = img.size
        left, top, right, bottom = crop_params
        right = width - right
        bottom = height - bottom
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)

def get_crop_params(filename):
    if filename.startswith("그림_H2017_1"):
        return (350,500,350,400)  # left, top, right, bottom for 그림_H2017_1
    elif filename.startswith("그림_H2017_2"):
        return (400, 500, 500, 350)  # left, top, right, bottom for 그림_H2017_2
    else:
        return (0,0,0,0)  # default values for other files

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{filename}")
            crop_params = get_crop_params(filename)
            crop_image(input_path, output_path, crop_params)
            print(f"Processed: {filename} with params: {crop_params}")

# 설정
input_folder = "Figure"
output_folder = "Figure_Cropped"

# 이미지 처리 실행
process_images(input_folder, output_folder)

print("모든 이미지 처리 완료")

