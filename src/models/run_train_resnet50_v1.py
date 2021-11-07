import os
import time
import csv
import shutil
import boto3
import numpy as np
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet50
from PIL import ImageFile, Image, ImageEnhance
#from src.preprocessing.data_augmentation import contrast_all_examples, horizontal_flip_all_examples, saturate_all_examples, brighten_all_examples, sharpen_all_examples
#from src.navigation import get_train_exterior_path, get_models_path, get_train_path, get_data_path
ImageFile.LOAD_TRUNCATED_IMAGES = True

# NAV
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

# model

def onehot_encode(classes, class_indices):
    """

    :param classes:
    :param class_indices:
    :return:
    """
    # one hot encode
    onehot_encoded = list()
    for value in classes:
        letter = [0 for _ in range(len(class_indices))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return np.array(onehot_encoded)

def load_train(img_height, img_width, train_path):
    """

    :param img_height:
    :param img_width:
    :param train_path:
    :return:
    """
    labels = os.listdir(train_path)
    labels = [p for p in labels if not p.endswith('jpg')]
    num = 0
    label2id = {}
    for label in labels:
        label2id[label] = num
        num += 1
    id2label = {v: k for k, v in label2id.items()}
    print("label2id : ", label2id)

    train_img = []
    train_label = []
    for label in labels:
        label_path = os.path.join(train_path, label)
        label_images = os.listdir(label_path)

        for i in range(len(label_images)):
            print(label_images[i])
            temp_img = image.load_img(os.path.join(label_path, label_images[i]), target_size=(img_height, img_width))
            temp_img = image.img_to_array(temp_img)

            train_img.append(temp_img)
            train_label.append(label2id[label])

    train_img = np.array(train_img, dtype='uint8')
    X_train = preprocess_input(train_img)
    train_label = np.array(train_label)
    y_train = onehot_encode(train_label, label2id)

    return X_train, y_train, label2id, id2label

def resnet50_model(num_classes):
    """
    :param num_classes:
    :return:
    """
    model = ResNet50(weights='imagenet', pooling='avg', include_top=False)
    x = Dropout(0.5)(model.output)
    x = Dense(num_classes, kernel_regularizer='l2')
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(model.input, x)
    
    # Train all layers
    for layer in model.layers:
        layer.trainable = True

    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC()])
    return model

# processing

def refactor_into_label_directories():
    """
    Refactors img data from hotel directories into label directories.
    :return: void
    """
    os.makedirs(os.path.join(get_train_path(), "exterior2"), exist_ok=True)
    count = 1
    hotel_dirs = os.listdir(get_train_exterior_path())
    for hotel_dir in hotel_dirs:
        hotel_files = os.listdir(os.path.join(get_train_exterior_path(), hotel_dir))
        for file in hotel_files:
            if(os.path.splitext(file)[1][1:] == "csv"):
                continue
            with open(os.path.join(get_train_exterior_path(), hotel_dir,
                                   hotel_files[len(hotel_files)-1])) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=",")
                for star in csvreader:
                    shutil.copy(os.path.join(get_train_exterior_path(), hotel_dir, file),
                                os.path.join(get_train_path(), "exterior2", str(star[1]) + "star"))
                    print("count: " + str(count))
                    count += 1
        os.remove(os.path.join(get_train_exterior_path(), hotel_dir))

def download_from_aws(num_images):
    s3 = boto3.resource('s3')
    corrupted = os.listdir(os.path.join(get_data_path(), "Corrupted"))
    count = 0
    labeled_exterior_images = s3.Bucket('labeled-exterior-images')
    for object_sum in labeled_exterior_images.objects.filter(Prefix=""):
        if(count == num_images):
            break
        if(corrupted.count(os.path.basename(object_sum.key)) != 0):
            continue
        os.system("aws s3 sync s3://labeled-exterior-images/" + os.path.dirname(object_sum.key) + " " + os.path.join(get_train_exterior_path(), os.path.dirname(object_sum.key)))
        print("num images downloaded: " + str(count))
        count +=1

# augmentation

def horizontal_flip_all_examples(examples_path):
    print("horizontal flipping...")
    examples = os.listdir(examples_path)
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        data = np.asarray(img)
        data = np.fliplr(data)
        img2 = Image.fromarray(data)
        img2name = os.path.splitext(example)[0] + "_horizontal_flipped" + ".png" 
        img2.save(os.path.join(examples_path, img2name))

def brighten_all_examples(examples_path):
    examples = os.listdir(examples_path)
    print("brightening...")
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        brightness_filter = ImageEnhance.Brightness(img)
        new_img = brightness_filter.enhance(1.1)
        new_img_name = os.path.splitext(example)[0] + "_brightened" + ".png" 
        new_img.save(os.path.join(examples_path, new_img_name))        

def sharpen_all_examples(examples_path):
    examples = os.listdir(examples_path)
    print("sharpening...")
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        sharpen_filter = ImageEnhance.Sharpness(img)
        new_img = sharpen_filter.enhance(1.1)
        new_img_name = os.path.splitext(example)[0] + "_sharpened" + ".png" 
        new_img.save(os.path.join(examples_path, new_img_name))       

def contrast_all_examples(examples_path):
    examples = os.listdir(examples_path)
    print("contrasting...")
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        filter = ImageEnhance.Contrast(img)
        new_img = filter.enhance(2)
        new_img_name = os.path.splitext(example)[0] + "_contrasted" + ".png" 
        new_img.save(os.path.join(examples_path, new_img_name))   

def saturate_all_examples(examples_path):
    examples = os.listdir(examples_path)
    print("saturating...")
    for example in examples:
        img = Image.open(os.path.join(examples_path, example))
        sat_filter = ImageEnhance.Color(img)
        new_img = sat_filter.enhance(3)
        new_img_name = os.path.splitext(example)[0] + "_saturated" + ".png" 
        new_img.save(os.path.join(examples_path, new_img_name))           

def augment_data(star_folder):
    examples_path = os.path.join(get_train_exterior_path(), star_folder)
    contrast_all_examples(examples_path)
    brighten_all_examples(examples_path)
    if(star_folder == "5star" or star_folder == "2star"):
        saturate_all_examples(examples_path)
    if(star_folder == "5star"):
        horizontal_flip_all_examples(examples_path)
        sharpen_all_examples(examples_path)

if __name__ == '__main__':
    to_augment = False
    os.makedirs(os.path.join(get_data_path(), "models"), exist_ok=True)
    download = input("Would you like to download images from AWS S3? Y/N: ")
    if(download == "N"):
        zip_downloaded = input("Have you downloaded images already from exterior.zip? Y/N: ")
        if(zip_downloaded == "N"):
            with ZipFile(os.path.join(get_train_path(), "exterior.zip"), 'r') as zipObj:
                zipObj.extractall(path=get_train_path())
            to_augment = True
    elif(download == "Y"):
        num_images = input("How many images would you like to download from AWS S3? Num images (integer): ")
        download_from_aws(int(num_images))

    if(to_augment == True):
        # Augment 4 star and 5 star and 2 star
        augment_data("4star")
        augment_data("5star")
        augment_data("2star")

    b_start = time.time()
    train_path = get_train_exterior_path()
    refactored = True
    if refactored == False:
        refactor_into_label_directories()
    model_path = os.path.join(get_models_path(), 'resnet50_ResNet50_v1.h5')

    img_height = 225
    img_width = 300
    batch_size = 32
    epochs = 100

    X_train, y_train, label2id, id2label = load_train(img_height, img_width, train_path)
    num_classes = len(label2id)

    model = resnet50_model(num_classes)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   verbose=1,
                                   restore_best_weights=True,
                                   patience=7)

    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[checkpointer],
                        verbose=1)

    print("Total used time : {} s.".format(time.time()-b_start))
