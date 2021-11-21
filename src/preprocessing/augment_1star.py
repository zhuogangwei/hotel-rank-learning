import os
import shutil
from PIL import Image, ImageEnhance
from src.navigation import get_train_exterior_path

def move_data_from_temp(examples_path, temp_folder):
    filenames = os.listdir(os.path.join(examples_path, temp_folder))
    for filename in filenames:
        shutil.move(os.path.join(examples_path, temp_folder, filename), examples_path)
    os.removedirs(os.path.join(examples_path, temp_folder))

def sharpen_all_examples(examples_path):
    examples = os.listdir(examples_path)
    os.makedirs(os.path.join(examples_path, "temp_sharpened"), exist_ok=True)
    print("sharpening...")
    for example in examples:
        _, ext = os.path.splitext(example)
        if(ext):
            img = Image.open(os.path.join(examples_path, example))
            sharpen_filter = ImageEnhance.Sharpness(img)
            new_img = sharpen_filter.enhance(1.2)
            new_img_name = os.path.splitext(example)[0] + "_sharpened3" + ".jpg"
            new_img.save(os.path.join(examples_path, "temp_sharpened", new_img_name))

def contrast_all_examples(examples_path):
    examples = os.listdir(examples_path)
    os.makedirs(os.path.join(examples_path, "temp_contrasted"), exist_ok=True)
    print("contrasting...")
    for example in examples:
        _, ext = os.path.splitext(example)
        if(ext):
            img = Image.open(os.path.join(examples_path, example))
            filter = ImageEnhance.Contrast(img)
            new_img = filter.enhance(1.3)
            new_img_name = os.path.splitext(example)[0] + "_contrasted3" + ".jpg"
            new_img.save(os.path.join(examples_path, "temp_contrasted", new_img_name))

if __name__ == "__main__":
    examples_path = os.path.join(get_train_exterior_path(), "1star")
    sharpen_all_examples(examples_path)
    contrast_all_examples(examples_path)
    move_data_from_temp(examples_path, "temp_sharpened")
    move_data_from_temp(examples_path, "temp_contrasted")
