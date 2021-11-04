import os
import csv
import requests
import boto3
from io import BytesIO

ACCESS_KEY = 'AKIAYDII5252MFKGD374'
SECRET_KEY = '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'

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
        img_url_data_path = os.path.join(img_url_data_path, img_url_files[i])
        print("Image Part " + str(i + 1))
        with open(img_url_data_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            count = 0

            # skip header
            next(csvreader)

            # upload
            for row in csvreader:
                count += 1
                if row[3] != '6' and row[3] != '11':
                    # only upload exterior images
                    continue

                # extract img from URL and save to memory
                img_url = row[1]
                img_name = os.path.basename(img_url)
                response = requests.get(img_url, stream=True)

                # upload to S3
                bucket = s3.Bucket(name='exterior-images')
                bucket.upload_fileobj(BytesIO(response.content), img_name)
                print("Uploading image " + img_name + " to S3 from row " + str(count) +"...")


if __name__ == "__main__":
    # call main
    main()
