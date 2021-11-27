"""
Scrapes hotel images across CPU cores given the star category.
"""
import os
import shutil
import requests
import time
from dask import dataframe as df
from multiprocessing import Pool, cpu_count
from src.utils import get_train_path

def download_image(url, hotelid, star_number):
    print("downloading ", url, "...")
    img_url_exterior = os.path.join(get_train_path(), "img_ext")
    star_folder = star_number + "star"
    img_name = str(hotelid) + "_" + os.path.basename(url)
    response = requests.get(url, stream=True)
    file = open(img_name, "wb")
    file.write(response.content)
    file.close()
    shutil.move(img_name, os.path.join(img_url_exterior, star_folder, img_name))

if __name__ == "__main__":
    start = time.time()
    img_url_exterior = os.path.join(get_train_path(), "imageURL-exterior")
    star_number = input("Num stars to download for: ")
    print("Star: ", star_number)
    star_folder = star_number + "star"
    csv_name = star_number + "star.csv"

    img_star_df = df.read_csv(os.path.join(img_url_exterior, csv_name))

    imageURLs = list(img_star_df['imageURL'])
    hotelids = list(img_star_df['masterhotelid'])
    star_list = [star_number] * len(imageURLs)
    print("There are {} CPUs on this machine ".format(cpu_count()))
    pool = Pool(cpu_count())
    pool.starmap(download_image, zip(imageURLs, hotelids, star_list))

    print("Run time: ", time.time()-start)

