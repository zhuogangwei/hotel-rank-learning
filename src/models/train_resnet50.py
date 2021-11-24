import os
import time
import csv
#import pandas as pd
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
from PIL import ImageFile
from src.navigation import get_train_exterior_path, get_models_path, get_train_path, get_data_path
from src.preprocessing.augment_image import augment_data

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def load_images(img_height, img_width, train_path):
    """

    :param img_height:
    :param img_width:
    :param train_path:
    :return:
    """
    labels = os.listdir(train_path)
    #label are 1star, ..., 5star. Image files are group into 5 folders, with folder name = star number 
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
            #print(label_images[i])
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
    #x = Dense(num_classes, kernel_regularizer='l2')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(model.input, x)
    
    # Train all layers
    for layer in model.layers:
        layer.trainable = True

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



if __name__ == '__main__':

    # User-prompted data download
    to_augment = False
    refactored = True

    os.makedirs(os.path.join(get_data_path(), "models"), exist_ok=True)
    download = input("Would you like to download images from AWS S3? Y/N: ")
    if(download == "N"):
        zip_downloaded = input("Have you downloaded images already from exterior.zip? Y/N: ")
        if(zip_downloaded == "N"):
            with ZipFile(os.path.join(get_train_path(), "exterior.zip"), 'r') as zipObj:
                zipObj.extractall(path=get_train_path())
            # to_augment = True
    elif(download == "Y"):
        num_images = input("How many hotels would you like to download images for from AWS S3? Num images (integer): ")
        download_from_aws(int(num_images))
        refactored = False

    # Data augmentation (skipped for milestone)
    if(to_augment == True):
        # Augment 4 star and 5 star and 2 star
        augment_data("1star")
        augment_data("3star")
        augment_data("4star")
        augment_data("5star")

    # training begin
    b_start = time.time()
    train_path = get_train_exterior_path()
    if refactored == False:
        refactor_into_label_directories()
    model_path = os.path.join(get_models_path(), 'resnet50_ResNet50_v1.h5')

    img_height = 225
    img_width = 300
    batch_size = 32
    epochs = 100

    X, Y, label2id, id2label = load_images(img_height, img_width, train_path)
    num_classes = len(label2id)

    model = resnet50_model(num_classes)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   verbose=1,
                                   restore_best_weights=True,
                                   patience=7)

    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
    history = model.fit(X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[checkpointer],
                        verbose=1)

    print("Total used time : {} s.".format(time.time()-b_start))