import os
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
#from src.navigation import get_train_exterior_path, get_data_path

def get_train_exterior_path():
    """
    Return the path to exterior training data.
    :return:
    """
    os.chdir("../../data/train/exterior")
    exterior_path = os.path.join(os.getcwd())
    print(exterior_path)
    os.chdir("../../../src/models")

    return exterior_path

def get_data_path():
    """
    Return the path to exterior training data.
    :return:
    """
    os.chdir("../../data/")
    data_path = os.path.join(os.getcwd())
    print(data_path)
    os.chdir("../src/models")

    return data_path

def horizontal_flip_all_examples(examples_path):
    examples = os.listdir(examples_path)
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        data = np.asarray(img)
        data = np.fliplr(data)
        img2 = Image.fromarray(data)
        img2name = os.path.basename(example).split['.'][0] + "_horizontal_flipped" + ".png" 
        img2.save(os.path.join(examples_path, img2name))

def brighten_all_examples(examples_path):
    examples = os.listdir(examples_path)
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        brightness_filter = ImageEnhance.Brightness(img)
        new_img = img.brightness_filter(1.1)
        new_img_name = os.path.basename(example).split['.'][0] + "_brightened" + ".png" 
        new_img.save(os.path.join(examples_path, new_img_name))        

def sharpen_all_examples(examples_path):
    examples = os.listdir(examples_path)
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        sharpen_filter = ImageEnhance.Sharpness(img)
        new_img = img.sharpen_filter(1.1)
        new_img_name = os.path.basename(example).split['.'][0] + "_sharpened" + ".png" 
        new_img.save(os.path.join(examples_path, new_img_name))       

def contrast_all_examples(examples_path):
    examples = os.listdir(examples_path)
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        contrast_filter = ImageEnhance.Contrast(img)
        new_img = img.contrast_filter(2)
        new_img_name = os.path.basename(example).split['.'][0] + "_contrasted" + ".png" 
        new_img.save(os.path.join(examples_path, new_img_name))   

def saturate_all_examples(examples_path):
    examples = os.listdir(examples_path)
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        sat_filter = ImageEnhance.Color(img)
        new_img = img.sat_filter(3)
        new_img_name = os.path.basename(example).split['.'][0] + "_saturated" + ".png" 
        new_img.save(os.path.join(examples_path, new_img_name))           

### DATA ANALYSIS ###
def class_analysis_before_augmentation():
    # count all elements in all classes
    num_examples_per_class = {}
    classes = os.listdir(get_train_exterior_path())
    for i in range(len(classes)):
        num_examples_per_class.update({i : len(os.listdir(os.path.join(get_train_exterior_path(), classes[i]))) })
    width = 1.0
    os.makedirs(os.path.join(get_data_path(), "data_analysis"), exist_ok=True)
    plt.bar(num_examples_per_class.keys(), num_examples_per_class.values(), width, color='g') 
    plt.title("Example Star Class Distribution (Before Augmentation)")
    plt.xlabel("Classes (# stars)")
    plt.ylabel("Number of Examples")
    print("saving class distribution...")
    plt.savefig(os.path.join(get_data_path(), "data_analysis", "class_distribution.png"))

if __name__ == "__main__":
    class_analysis_before_augmentation()
