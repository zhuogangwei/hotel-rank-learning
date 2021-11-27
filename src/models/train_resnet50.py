import csv
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet50
from PIL import ImageFile
#from src.utils import get_train_exterior_path, get_models_path, get_train_path, get_data_path, star_onehot_encode, is_corrupted
#from src.preprocessing.augment_image import augment_data

ImageFile.LOAD_TRUNCATED_IMAGES = True
results = []

def star_onehot_encode(stars):
    """

    :param stars: 1D array
    :return: one-hot encoded star ratings
    """
    # one hot encode
    num_class = 5 #from 1 star to 5 stars
    onehot_encoded = list()
    for star in stars:
        encoded = np.zeros(num_class)
        encoded[star-1] = 1
        onehot_encoded.append(encoded)

    return np.array(onehot_encoded)

def is_corrupted(filename, star):
    corrupted_path = get_corrupted_path()
    corrupted_list = []
    file = open(os.path.join(corrupted_path, star+"star"+".csv"), "r")
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        corrupted_list.append(row)
    if filename in corrupted_list[0]:
        return True
    return False

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

def get_corrupted_path():
    """
    Return the path to directory containing csvs of corrupted data.
    :return: corrupted
    """
    os.chdir("../../data/train/corrupted")
    corrupted = os.path.join(os.getcwd())
    os.chdir("../../../src/models")

    return corrupted

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

def load_images(img_height, img_width, train_path, skip_deserialize=False):
    """
    :param img_height:
    :param img_width:
    :param train_path:
    :return:
    """
    labels = os.listdir(train_path)
    #label are 1star, ..., 5star. Image files are group into 5 folders, with folder name = star number 
    labels = [p for p in labels if p.endswith('star')]

    hotelid_image_mapping = pd.DataFrame(columns=['hotelid', 'image_serialized', 'star'])

    idx=0
    for label in labels:
        #process_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        #with 
        #hotel_image_mapping 
        #args = )
        label_path = os.path.join(train_path, label)
        image_filenames = os.listdir(label_path)
        temp_star = label[0] # first char of '5star' is 5

        for image_filename in image_filenames:
            if(is_corrupted(image_filename, temp_star) == True):
                print("Skipping corrupted file ", image_filename, " from ", temp_star, " stars...")
                continue            
            temp_img = image.load_img(os.path.join(label_path, image_filename), target_size=(img_height, img_width))
            # image serialization
            temp_img = image.img_to_array(temp_img).astype('uint8').tobytes()
            temp_hotelid = int(image_filename[0 : image_filename.find('_')])
            new_row = pd.DataFrame([[temp_hotelid, temp_img, temp_star]], columns=hotelid_image_mapping.columns, index=[idx])
            hotelid_image_mapping = hotelid_image_mapping.append(new_row)
            idx += 1
            #train_img.append(temp_img)
            #train_label.append(label2id[label])
    
    # shuffle image orders
    hotelid_image_mapping = hotelid_image_mapping.sample(frac=1)
    
    if (skip_deserialize==True):
        return None, None, hotelid_image_mapping
    else:
        X_train, y_train = deserialize_image(hotelid_image_mapping, img_height, img_width)
        return X_train, y_train, hotelid_image_mapping

def deserialize_image(hotelid_image_mapping, img_height, img_width):

    train_img = list()
    train_label = []
    
    #train_label = np.array(train_label)
    train_label = hotelid_image_mapping['star'].to_numpy(dtype='uint8', copy = True)
    y_train = star_onehot_encode(train_label)
    
    #image deserialization
    print("image deserialization...")
    num_images = hotelid_image_mapping['image_serialized'].count()
    for idx in range(num_images):
        temp_img = hotelid_image_mapping.at[idx, 'image_serialized']
        temp_deserialized_img = np.frombuffer(temp_img, dtype='uint8').reshape(img_height, img_width, 3)
        train_img.append(temp_deserialized_img)
        #after deserizlization of each image, drop the serialized image to release memory
        hotelid_image_mapping.drop(index=idx, inplace=True)

    train_img = np.array(train_img, dtype='uint8')
    print("preprocess input...")
    X_train = preprocess_input(train_img)

    #train_label = np.array(train_label)
    train_label = hotelid_image_mapping['star'].to_numpy(dtype='uint8', copy = True)
    y_train = star_onehot_encode(train_label)
    return X_train, y_train

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
    
    # Train last few layers
    for layer in model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC()])
    return model

# processing

if __name__ == '__main__':

    os.makedirs(os.path.join(get_data_path(), "models"), exist_ok=True)
    
    # training begin
    b_start = time.time()
    train_path = get_train_exterior_path()
    model_path = os.path.join(get_models_path(), 'resnet50_ResNet50_v1.h5')

    img_height = 225
    img_width = 300
    batch_size = 32
    epochs = 100

    X, Y, _ = load_images(img_height, img_width, train_path)
    breakpoint()
    num_classes = 5 # five star categories

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
