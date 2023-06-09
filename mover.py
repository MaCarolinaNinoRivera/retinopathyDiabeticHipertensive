import pandas as pd #For reading csv files.
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image

# method that reads a csv file and converts it to a list
def create_list():
    listImages = []
    path = "./trainLabels_or.csv"

    train_df = pd.read_csv(f"{path}", names=['image','level'])

    print(f'No.of.training_samples: {len(train_df)}')
    for i, row in train_df.iterrows():
        listImages.append(row['image'] + "," + row['level'])
    return listImages
# method that receives the initial path and the path where you want to move the images
def moved_images(input_path_folder, output_path_folder):
    # listExcell = create_list()
    listExcell = []
    """
    Uses multiprocessing to make it fast
    """
    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)
    
    jobs = [
        (file, input_path_folder, output_path_folder, listExcell)
        for file in os.listdir(input_path_folder)
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))
# method that reads the list and finds to which class the image 
# belongs to be saved in the selected path according to the class
def searchCompletePath(img_file2, listExcell):
    restAddress = ''
    for i, img in enumerate(listExcell):
        # Split the name into the image and disease level fields
        name = img.split(',')
        if(img_file2 in name[0]):
            if(name[1] == "0"):
                restAddress = 'clase0/'
                break
            elif(name[1] == "1"):
                restAddress = 'clase1/'
                break
            elif(name[1] == "2"):
                restAddress = 'clase2/'
                break
            elif(name[1] == "3"):
                restAddress = 'clase3/'
                break
            else:
                restAddress = 'clase4/'
                break
    return restAddress
# Method that finally saves the image to the specified path and found folder
def save_single(args):
    img_file, input_path_folder, output_path_folder, listExcell = args
    img_file2 = img_file.replace(".jpeg", "")
    # Find the class to which the image belongs
    # nivelFinal = searchCompletePath(img_file2, listExcell)
    nivelFinal = ''
    image_original = Image.open(os.path.join(input_path_folder, img_file))
    image_original.save(os.path.join(output_path_folder + nivelFinal + img_file))

if __name__ == "__main__":
    moved_images("G:/Mi unidad/Colab Notebooks/retinopathy/Diabetic/Test/resized_650/", "C:/Users/carol/OneDrive/Documentos/Sergio/KaggleRetinopathy/baseLine/Data/Test/resized_650/")  