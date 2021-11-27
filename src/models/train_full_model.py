import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU
from tensorflow.keras.applications.resnet import ResNet50
from sklearn import metrics
from sklearn.model_selection import train_test_split
from src.utils import get_train_exterior_path
from train_resnet50 import load_images, deserialize_image, resnet50_model
from train_structured import load_metadata, DNN_model

def align_model_inputs(hotelid_image_mapping, metaX, meta_hotelids, img_height, img_width):
    """
    align the inputs of the CNN(images) and DNN(metadata) models, such that two inputs have identical sample size
    and the samples of the same index position across two input arrays carry the same hotelid.
    Referential relationship is that metadata:images = 1:many
    """
    #inner join image and metadata on hotelid
    metaX = pd.concat([metaX, meta_hotelids.astype('int32')], axis=1)
    df_merged = hotelid_image_mapping.merge(metaX, how='inner', left_on='hotelid', right_on='hotelid')
        
    #split train and test sets
    df_merged_train, df_merged_val = train_test_split(df_merged, test_size=0.2, random_state=0)
    df_merged_train.reset_index(inplace=True)
    df_merged_val.reset_index(inplace=True)
    
    #after the alignment join, split the merged dataset back into image and metadata
    metaX_train = df_merged_train[metaX.columns].copy(deep=True).astype('float16')
    metaX_val = df_merged_val[metaX.columns].copy(deep=True).astype('float16')
    metaX_train = np.nan_to_num(np.asarray(metaX_train.drop(['hotelid'], axis = 1)))
    metaX_val = np.nan_to_num(np.asarray(metaX_val.drop(['hotelid'], axis = 1)))
    
    #after the alignment join, split the merged dataset back into image and metadata
    imageX_train, Y_train = deserialize_image(df_merged_train, img_height, img_width)
    imageX_val, Y_val = deserialize_image(df_merged_val, img_height, img_width)

    return imageX_train, metaX_train, imageX_val, metaX_val, Y_train, Y_val
    
if __name__ == '__main__':
    img_height = 187
    img_width = 250 # get this closer to 224
    channels = 3
    batch_size = 32
    DNN_epochs = 30
    CNN_epochs = 1
    num_classes = 5
    
    train_path = get_train_exterior_path()
    
    _, _, hotelid_image_mapping = load_images(img_height, img_width, train_path, skip_deserialize=True)
    metaX, _, meta_hotelids = load_metadata("hotel_meta_processed.csv")
    imageX_train, metaX_train, imageX_val, metaX_val, Y_train, Y_val = align_model_inputs(hotelid_image_mapping, metaX, meta_hotelids, img_height, img_width)
    
    input_CNN = Input(shape=(img_height, img_width, channels))
    input_DNN = Input(shape=(metaX_train.shape[1]))
    
    CNN_model = ResNet50(weights='imagenet', pooling='avg', include_top=False)
    CNN_dense1 = Dense(512, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal')(CNN_model.output)
    CNN_dense2 = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal')(CNN_dense1)
    CNN_dense3 = Dense(32, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal')(CNN_dense2)
    CNN_dense4 = Dense(8, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal')(CNN_dense3)
    
    dnn_model = DNN_model((metaX.shape[1],))
    
    # Train last few layers
    for layer in CNN_model.layers[:-19]:
        layer.trainable = False
    
    """
    CNN_model.fit(imageX_train, Y_train,
                  validation_data=(imageX_val, Y_val),
                  epochs=CNN_epochs, batch_size=batch_size, shuffle=False, verbose=1)
    """
    
    # Concatenate
    concat = tf.keras.layers.Concatenate()([CNN_dense4, dnn_model])

    # output layer input_shape=(None, concat.shape[-1])
    output = Dense(units=num_classes, activation='softmax')(concat)
    
    full_model = tf.keras.Model(inputs=[input_CNN, input_DNN], outputs=[output])
    print(full_model.summary())
    
    full_model.fit([metaX_train, imageX_train], Y_train,
                  validation_data=([metaX_val, imageX_val], Y_val),
                  epochs=CNN_epochs, batch_size=batch_size, shuffle=False, verbose=1)
    