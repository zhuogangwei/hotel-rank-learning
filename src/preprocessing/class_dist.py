import os
from matplotlib import pyplot as plt
from src.navigation import get_train_exterior_path, get_data_path

### DATA ANALYSIS ###
def class_analysis_before_augmentation():
    # count all elements in all classes
    num_examples_per_class = {}
    classes = os.listdir(get_train_exterior_path())
    for i in range(len(classes)):
        num_examples_per_class.update({i+1 : len(os.listdir(os.path.join(get_train_exterior_path(), classes[i]))) })
    width = 1.0
    os.makedirs(os.path.join(get_data_path(), "data_analysis"), exist_ok=True)
    plt.bar(num_examples_per_class.keys(), num_examples_per_class.values(), width, color='g') 
    plt.title("Star Rating Class Distribution")
    plt.xlabel("Classes (# stars)")
    plt.ylabel("Number of Examples")
    print("saving class distribution...")
    plt.savefig(os.path.join(get_data_path(), "data_analysis", "class_distribution.png"))

if __name__ == "__main__":
    class_analysis_before_augmentation()
