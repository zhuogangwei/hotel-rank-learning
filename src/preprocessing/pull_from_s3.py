import os
import csv
import shutil
import numpy as np
import requests
import time
from dask import dataframe as df
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, cpu_count
#from src.utils import get_train_path

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

def download_image(url, hotelid):
    print("downloading ", url, "...")
    img_url_exterior = os.path.join(get_train_path(), "img_ext")
    star_number = str(5) # change based on class
    star_folder = star_number + "star"
    img_name = hotelid + "_" + os.path.basename(url)
    response = requests.get(url, stream=True)
    file = open(img_name, "wb")
    file.write(response.content)
    file.close()
    shutil.move(os.path.basename(url), os.path.join(img_url_exterior, star_folder, img_name))

if __name__ == "__main__":
    start = time.time()
    img_url_exterior = os.path.join(get_train_path(), "exterior")
    star_number = input("Num stars to download for: ")
    print("Star: ", star_number)
    star_folder = star_number + "star"
    csv_name = star_number + "star.csv"

    img_star_df = df.read_csv(os.path.join(img_url_exterior, csv_name))

    records = img_star_df.to_records(index=False)
    inputs = list(records)
    print("There are {} CPUs on this machine ".format(cpu_count()))
    with Pool(processes=cpu_count()):
        res = pool.starmap(download_image, inputs)
    
    print("Run time: ", time.time()-start)


    #img_urls = list(img_star_df.imageURL)
    #print("There are {} CPUs on this machine ".format(cpu_count()))
    #pool = Pool(cpu_count())
    #results = pool.map(download_url, img_urls)
    #print("Run time: ", time.time()-start)

