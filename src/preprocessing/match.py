########################################################################
## Uses a Hash Map to add Hotelid to Image name for combined training ##
########################################################################

import os
from src.navigation import get_train_path, get_train_exterior_path
from dask import dataframe as df

if __name__ == "__main__":
    image_parts = ["image_part1.csv", "image_part2.csv", "image_part3.csv"]
    for image_part in image_parts:
        print(image_part)
        img_part_df = df.read_csv(os.path.join(get_train_path(), image_part))
        img_dict = dict(zip(img_part_df.url, img_part_df.hotelid))
        for i in range(5):
            print("Star:", str(i+1))
            dir = str(i+1) + "star"
            examples = os.listdir(os.path.join(get_train_exterior_path(), dir))
            for example in examples:
                e = example.partition("R5")[0]
                res = [val for key, val in img_dict.items() if e in key]
                if res:
                    id = res[0]
                    print("RENAME", example, " with hotelid:", id)
                    os.rename(os.path.join(get_train_exterior_path(), dir, example),
                            os.path.join(get_train_exterior_path(), dir, str(id) + "_" + example))
