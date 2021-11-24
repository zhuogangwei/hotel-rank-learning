# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:17:35 2021

@author: Bo Xian Ye
"""
import os
import numpy as np
from PIL import Image, ImageEnhance
import shutil
from src.navigation import get_train_exterior_path

# augmentation
def horizontal_flip_all_examples(examples_path):
    print("flipping examples horizontally...")
    examples = os.listdir(examples_path)
    os.makedirs(os.path.join(examples_path, "temp_flipped"), exist_ok=True)
    for example in examples:
        _, ext = os.path.splitext(example)
        if(ext):
            img = Image.open(os.path.join(examples_path, example))
            data = np.asarray(img)
            data = np.fliplr(data)
            img2 = Image.fromarray(data)
            img2name = os.path.splitext(example)[0] + "_horizontal_flipped" + ".jpg"
            img2.save(os.path.join(examples_path, "temp_flipped", img2name))

def brighten_all_examples(examples_path):
    examples = os.listdir(examples_path)
    print("brightening...")
    os.makedirs(os.path.join(examples_path, "temp_brightened"), exist_ok=True)
    for example in examples:
        _, ext = os.path.splitext(example)
        if(ext):
            img = Image.open(os.path.join(examples_path, example))
            brightness_filter = ImageEnhance.Brightness(img)
            new_img = brightness_filter.enhance(1.1)
            new_img_name = os.path.splitext(example)[0] + "_brightened" + ".jpg"
            new_img.save(os.path.join(examples_path, "temp_brightened", new_img_name))

def sharpen_all_examples(examples_path):
    examples = os.listdir(examples_path)
    os.makedirs(os.path.join(examples_path, "temp_sharpened"), exist_ok=True)
    print("sharpening...")
    for example in examples:
        _, ext = os.path.splitext(example)
        if(ext):
            img = Image.open(os.path.join(examples_path, example))
            sharpen_filter = ImageEnhance.Sharpness(img)
            new_img = sharpen_filter.enhance(1.1)
            new_img_name = os.path.splitext(example)[0] + "_sharpened" + ".jpg"
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
            new_img = filter.enhance(2)
            new_img_name = os.path.splitext(example)[0] + "_contrasted" + ".jpg"
            new_img.save(os.path.join(examples_path, "temp_contrasted", new_img_name))

def saturate_all_examples(examples_path):
    examples = os.listdir(examples_path)
    os.makedirs(os.path.join(examples_path, "temp_saturated"), exist_ok=True)
    print("saturating...")
    enhance_factors = [0.2, 0.1, 0.25, 0.3, 0.4]
    for enhance_factor in enhance_factors:
        for example in examples:
            _, ext = os.path.splitext(example)
            if(ext):
                img = Image.open(os.path.join(examples_path, example))
                sat_filter = ImageEnhance.Color(img)
                while enhance_factor < 4:
                    new_img = sat_filter.enhance(enhance_factor)
                    new_img_name = os.path.splitext(example)[0] + "_saturated_" + str(enhance_factor) + ".jpg"
                    new_img.save(os.path.join(examples_path, "temp_saturated", new_img_name))
                    enhance_factor += 0.5

def move_data_from_temp(examples_path, temp_folder):
    filenames = os.listdir(os.path.join(examples_path, temp_folder))
    for filename in filenames:
        shutil.move(os.path.join(examples_path, temp_folder, filename), examples_path)
    os.removedirs(os.path.join(examples_path, temp_folder))

def augment_data(star_folder):
    examples_path = os.path.join(get_train_exterior_path(), star_folder)
    contrast_all_examples(examples_path)
    if(star_folder == "5star" or "1star"):
        brighten_all_examples(examples_path)
        sharpen_all_examples(examples_path)
    if(star_folder == "1star"):
        saturate_all_examples(examples_path)
        horizontal_flip_all_examples(examples_path)
        move_data_from_temp(examples_path, "temp_flipped")
        move_data_from_temp(examples_path, "temp_saturated")
    if(star_folder == "5star" or star_folder == "1star"):
        move_data_from_temp(examples_path, "temp_sharpened")
        move_data_from_temp(examples_path, "temp_brightened")
    move_data_from_temp(examples_path, "temp_contrasted")

