import csv
from shutil import move
from os import listdir
from PIL import Image

count = 0
corrupted_filenames = []
for filename in listdir('../../data/train/exterior/1star/'):
    if filename.endswith('.jpg'):
        try:
            img = Image.open("../../data/train/exterior/1star/" + filename) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print(count)
            count = count+1
            corrupted_filenames.append(filename)
            print('Bad file:', filename) # print out the names of corrupt files
with open('1star.csv', 'w') as file:
    write = csv.writer(file)
    write.writerow(corrupted_filenames)

move('1star.csv', '../../data/train/corrupted/1star.csv')