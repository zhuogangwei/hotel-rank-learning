import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from keras import metrics
from keras.applications.resnet import preprocess_input
from keras.models import Input, Model
from keras.layers import Lambda, GlobalAveragePooling2D, Dropout, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def onehot_encode(classes, class_indices):
    # one hot encode
    onehot_encoded = list()
    for value in classes:
        letter = [0 for _ in range(len(class_indices))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return np.array(onehot_encoded)


def load_train(img_height, img_width, train_path):
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
            temp_img = image.load_img(os.path.join(label_path, label_images[i]), target_size=(img_height, img_width))
            temp_img = image.img_to_array(temp_img)

            train_img.append(temp_img)
            train_label.append(label2id[label])

    train_img = np.array(train_img)
    X_train = preprocess_input(train_img)
    train_label = np.array(train_label)
    y_train = onehot_encode(train_label, label2id)

    return X_train, y_train, label2id, id2label


def resnet50_model(num_classes):
    model = ResNet50(weights='imagenet', pooling='avg', include_top=False)
    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    x = Dropout(0.3)(model.output)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(model.input, x)
    
    # To set the first 30 layers to non-trainable (weights will not be updated)
    for layer in model.layers[:-30]:
        layer.trainable = False

    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC()])
    return model


if __name__ == '__main__':
    b_start = time.time()
    train_path = './data/exterior'

    model_path = './data/models/resnet50_ResNet50_v1.h5'
    img_height = 225
    img_width = 300
    batch_size = 16
    epochs = 10

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