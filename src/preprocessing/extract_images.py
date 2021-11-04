import os
import csv
import requests
import boto3
from io import BytesIO
from src.navigation import get_img_url_data_directory

ACCESS_KEY = 'AKIAYDII5252MFKGD374'
SECRET_KEY = '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'


def upload_to_s3(temp_path, img_name, count):
    """
    Uploads the image specified to S3
    :param img_name:
    :return:
    """
    os.chdir("temp")
    os.system("aws s3 cp " + os.path.join(temp_path, img_name) + " s3://hotel-rating-images/")
    os.chdir("..")
    print("Uploading image " + img_name + " to S3 from row: " + str(count) +"...")

def main():
    # client application is s3
    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    # get file names into a list
    img_url_files = []
    img_url_data_path = get_img_url_data_directory()
    num_files = len(os.listdir(img_url_data_path))
    for i in range(num_files):
        filename = "image_part" + str(i+1) + ".csv"
        img_url_files.append(filename)

    # upload image to AWS
    for i in range(num_files):
        img_url_data_path = os.path.join(img_url_data_path, img_url_files[i+1])
        print("Image Part " + str(i + 1))
        with open(img_url_data_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            count = 0

            # skip header
            next(csvreader)

            # upload
            for row in csvreader:
                count += 1

                # retrieve img into memory
                img_url = row[1]
                img_name = os.path.basename(img_url)
                response = requests.get(img_url, stream=True)

                # upload to S3
                bucket = s3.Bucket(name='hotel-rating-images')
                bucket.upload_fileobj(BytesIO(response.content), img_name)
                print("Uploading image " + img_name + " to S3 from row: " + str(count) +"...")


if __name__ == "__main__":
    # call main
    main()
