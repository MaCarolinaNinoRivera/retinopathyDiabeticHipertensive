import os
import numpy as np
import cv2
from glob import glob # extract file paths and images
from tqdm import tqdm # progress bar
from albumentations import HorizontalFlip, VerticalFlip, CLAHE # for the augmentation

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "Data", "Level4", "*.jpeg")))

    return (train_x, train_x), (train_x, train_x)

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Replace \ by / """
        x = x.replace("\\", "/")
        y = y.replace("\\", "/")
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]
        # """ Reading image """
        x = cv2.imread(x, cv2.IMREAD_COLOR)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x)
            x1 = augmented["image"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x)
            x2 = augmented["image"]

            aug = CLAHE(p=0.6)
            augmented = aug(image=x)
            x3 = augmented["image"]            

            X = [x, x1, x2, x3]
            Y = [x, x1, x2, x3]

        else:
            X = [x]
            Y = [x]

        index = 0
        for i,m in zip(X,Y):
            i = cv2.resize(i, size)

            tmp_image_name = f"{name}_{index}.jpeg"

            image_path = os.path.join(save_path, "Augmentation4", tmp_image_name)

            cv2.imwrite(image_path, i)

            index += 1

if __name__ == "__main__":
    """ Seeding """

    """ Load the data """
    data_path = "C:/Users/carol/OneDrive/Documentos/Sergio/KaggleRetinopathy/baseLine/"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")

    """ Create directories to save the augmented data """
    create_dir("C:/Users/carol/OneDrive/Documentos/Sergio/KaggleRetinopathy/baseLine/Data/Augmentation4/")

    """ Data augmentation """
    augment_data(train_x, train_y, "C:/Users/carol/OneDrive/Documentos/Sergio/KaggleRetinopathy/baseLine/Data/", augment=True)
