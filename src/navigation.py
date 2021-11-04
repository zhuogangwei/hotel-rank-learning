"""
Navigate through directories and files.
"""
import os

"""PREPROCESSING"""
def get_img_url_data_directory():
    """
    Return the URL data directory.
    :return: img_url_data_path
    """
    # change directory
    os.chdir("../../url_data/image_urls")

    # get directory
    img_url_data_path = os.path.join(os.getcwd())
    print(img_url_data_path)

    os.chdir("../../src/preprocessing")

    return img_url_data_path

def get_temp_dir():
    """
    Return the temp directory which stores temporary labeled packages.
    :return: temp_data_path
    """
    # change directory
    os.chdir("../../temp")

    # get directory
    temp_data_path = os.path.join(os.getcwd())
    print(temp_data_path)

    os.chdir("../src/preprocessing")

    return temp_data_path

def get_structured_data_dir():
    """
    Return directory for structured data.
    :return: structured_data_path
    """
    # change directory
    os.chdir("../../url_data/structured_hotel_data")

    # get directory
    structured_data_path = os.path.join(os.getcwd())
    print(structured_data_path)

    os.chdir("../../src/preprocessing")

    return structured_data_path

"""TRAINING & INFERENCE"""
def get_train_path():
    """
    Return the path to training data directories.
    :return: train_path
    """
    os.chdir("../../data/train/")
    train_path = os.path.join(os.getcwd())
    print(train_path)
    os.chdir("../../src/models")

    return train_path

def get_models_path():
    """
    Return the models path which stores the model checkpoint at a frequency.
    :return: models_path
    """
    os.chdir("../../data/models/")
    models_path = os.path.join(os.getcwd())
    print(models_path)
    os.chdir("../../src/models")

    return models_path

def get_train_exterior_path():
    """
    Return the path to exterior training data.
    :return:
    """
    os.chdir("../../data/train/exterior")
    exterior_path = os.path.join(os.getcwd())
    print(exterior_path)
    os.chdir("../../src/models")

    return exterior_path
