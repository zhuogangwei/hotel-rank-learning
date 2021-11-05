import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def get_structured_data_path():
    """
    Return the path to structured data.
    :return: structured_data_path
    """
    os.chdir("../../data/url_data/structured_hotel_data")
    structured_data_path = os.path.join(os.getcwd())
    os.chdir("../../../src/models/")
    return structured_data_path

if __name__ == "__main__":
    raw_structured_data = pd.read_csv(os.path.join(get_structured_data_path(), "Hotel_metadata.csv"))
    # input
    x = raw_structured_data[["openyear", "numReviewers", "roomRating", "serviceRating",
                            "value4moneyRating", "locatioRating", "geoid", "roomquantity",
                            "minArea", "isStarValidated", "order_cnt", "roomnight_cnt",
                            "avg_room_price", "avg_person_price", "maxprice", "minprice",
                            "adr1", "adr2", "adr3", "adr4", "adr5", "gym", "executive_lounge",
                            "indoor_swimming_pool", "bathrobe", "laundry_service", "X24h_frontdesk",
                            "conference_hall", "luggage_storage", "roomcleaneddaily",
                            "outdoor_swimming_pool", "avg_person_price_bycity", "pic_hq_ratio",
                            "pic_apperance_ratio", "pic_public_ratio", "pic_meeting_ratio",
                            "pic_restaurant_ratio", "pic_leisure_ratio", "pic_service_ratio",
                            "renovationyear", "is_adr2_adjusted", "is_adr3_adjusted",
                            "is_adr4_adjusted", "is_adr5_adjusted"]]

    # output
    y = raw_structured_data["star"]

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)

    # define model & train
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(model.coef_)
    print(model.intercept_)
    pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])

    # inference
    predictions = model.predict(x_test)

    # metrics
    metrics.mean_absolute_error(y_test, predictions)
    metrics.mean_squared_error(y_test, predictions)
    np.sqrt(metrics.mean_squared_error(y_test, predictions))

