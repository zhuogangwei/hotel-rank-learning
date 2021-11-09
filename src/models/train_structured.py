import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def get_structured_data_path():
    """
    Return the path to structured data.
    :return: structured_data_path
    """
    os.chdir("../../data")
    structured_data_path = os.path.join(os.getcwd())
    os.chdir("../src/models")
    return structured_data_path

def run_linear_model(x_train, y_train, x_test):
    # define model & train
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(model.coef_)
    print(model.intercept_)
    pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])
    # inference
    predictions = np.round(model.predict(x_test))
    predictions[predictions>5] =5
    predictions[predictions<1] =1

    # metrics
    metrics.mean_absolute_error(y_test, predictions)
    metrics.mean_squared_error(y_test, predictions)
    np.sqrt(metrics.mean_squared_error(y_test, predictions))
    correct = predictions == y_test
    acc = sum(correct)/len(predictions)
    print('Test Accuracy: %.3f' % acc)

def run_DNN_model(x_train, y_train, x_test, y_test, epochs, batch_size):
    #one-hot encoding for labels
    encoder = OneHotEncoder(handle_unknown='ignore')
    y_train = pd.DataFrame(encoder.fit_transform(y_train.reshape(len(y_train),1)).toarray())
    y_test = pd.DataFrame(encoder.fit_transform(y_test.reshape(len(y_test),1)).toarray())
    
    # define model
    model = Sequential()
    model.add(Dense(81, activation='relu', kernel_initializer='he_normal', input_shape=(x_train.shape[1],)))
    model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(5, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # fit the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    # inference
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)


if __name__ == "__main__":
    raw_structured_data = pd.read_csv(os.path.join(get_structured_data_path(), "hotel_meta_processed.csv"))
    # input
    x1 = raw_structured_data[["openyear", "numReviewers", "roomRating", "serviceRating",
                            "value4moneyRating", "locatioRating", "roomquantity",
                            "minArea", "isStarValidated", "order_cnt", "roomnight_cnt",
                            "avg_room_price", "avg_person_price", "maxprice", "minprice",
                            "adr1", "adr2", "adr3", "adr4", "adr5", "avg_person_price_bycity", "pic_hq_ratio",
                            "pic_apperance_ratio", "pic_public_ratio", "pic_meeting_ratio",
                            "pic_restaurant_ratio", "pic_leisure_ratio", "pic_service_ratio",
                            "is_adr2_adjusted", "is_adr3_adjusted",
                            "is_adr4_adjusted", "is_adr5_adjusted"]]
    x2 = raw_structured_data[["renovationyear", "gym", "executive_lounge",
                            "indoor_swimming_pool", "bathrobe", "laundry_service", "X24h_frontdesk",
                            "conference_hall", "luggage_storage", "roomcleaneddaily", "outdoor_swimming_pool"]]
    
    encoder = OneHotEncoder(handle_unknown='error')
    x2_onehot = pd.DataFrame(encoder.fit_transform(x2).toarray())

    x = pd.concat([x1, x2_onehot], axis=1)

    # output
    y = raw_structured_data["star"]

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)

    #handling null(if any)
    x_train = np.nan_to_num(x_train)
    y_train = np.nan_to_num(y_train)
    x_test = np.nan_to_num(x_test)
    y_test = np.nan_to_num(y_test)

    #run_linear_model(x_train, y_train, x_test)
    run_DNN_model(x_train, y_train, x_test, y_test, 100, 32)
    
