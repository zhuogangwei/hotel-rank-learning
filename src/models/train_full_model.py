# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:31:36 2021

@author: Bo Xian Ye
"""
import pandas as pd
from src.navigation import get_train_exterior_path
from train_resnet50 import load_images, deserialize_image
from train_structured import load_metadata

def align_model_inputs(hotelid_image_mapping, metaX, meta_hotelids, img_height, img_width):
    """
    align the inputs of the CNN(images) and DNN(metadata) models, such that two inputs have identical sample size
    and the samples of the same index position across two input arrays carry the same hotelid.
    Referential relationship is that metadata:images = 1:many
    """
    #inner join image and metadata on hotelid
    metaX = pd.concat([metaX, meta_hotelids.astype('int32')], axis=1)
    df_merged = hotelid_image_mapping.merge(metaX, how='inner', left_index=True, right_on='hotelid')
    #print (df_merged.count())
    
    #after the alignment join, split the merged dataset back into two
    imageX, Y = deserialize_image(df_merged, img_height, img_width)
    metaX = df_merged[metaX.columns]
    metaX.drop(['hotelid'], axis = 1)

    return imageX, metaX, Y
    
if __name__ == '__main__':
    img_height = 225
    img_width = 300
    batch_size = 32
    epochs = 100
    
    train_path = get_train_exterior_path()
    
    _, _, hotelid_image_mapping = load_images(img_height, img_width, train_path, skip_deserialize=True)
    metaX, _, meta_hotelids = load_metadata("hotel_meta_processed.csv")
    imageX, metaX, Y = align_model_inputs(hotelid_image_mapping, metaX, meta_hotelids, img_height, img_width)