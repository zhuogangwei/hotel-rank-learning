import os
import shutil
import csv
from urllib.request import urlretrieve

def get_img_url_data_directory():
    """
    Return the URL data directory.
    :return: path to URL data directory
    """
    # change directory
    os.chdir("../../url_data/image_urls")

    # get directory
    img_url_data_path = os.path.join(os.getcwd())
    print(img_url_data_path)

    os.chdir("../../src/preprocessing")

    return img_url_data_path

def upload_to_s3(temp_path, img_name):
    """
    Uploads the image specified to S3
    :param img_name:
    :return:
    """
    os.chdir("temp")
    os.system("aws s3 cp " + os.path.join(temp_path, img_name) + " s3://hotel-rating-images/")
    os.chdir("..")
    print("Uploading image " + img_name + " to S3...")


def main():
    # get file names into a list
    img_url_files = []
    img_url_data_path = get_img_url_data_directory()
    num_files = len(os.listdir(img_url_data_path))
    for i in range(num_files):
        filename = "image_part" + str(i+1) + ".csv"
        img_url_files.append(filename)

    # upload image to AWS
    for i in range(num_files):
        img_url_data_path = os.path.join(img_url_data_path, img_url_files[i])
        with open(img_url_data_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")

            # skip header
            next(csvreader)

            # upload
            for row in csvreader:
                # temp path to retrieve img to
                temp_path = os.path.join(os.getcwd(), "temp")
                os.makedirs(temp_path, exist_ok=True)

                # retrieve img locally
                img_url = row[1]
                img_name = os.path.basename(img_url)
                os.chdir(temp_path)
                urlretrieve(img_url, os.path.join(img_name))
                os.chdir("..")

                # upload to S3
                upload_to_s3(temp_path, img_name)

                # clean up temporary folder with img
                shutil.rmtree("temp")


if __name__ == "__main__":
    # call main
    main()
