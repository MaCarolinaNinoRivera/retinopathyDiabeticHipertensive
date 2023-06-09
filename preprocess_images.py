import os
"""
Tries to remove unnecessary black borders around the images, and
"trim" the images to they take up the entirety of the image.
"""

import cv2
import numpy as np
from PIL import Image
import warnings
from multiprocessing import Pool
from tqdm import tqdm

def trim(im):
    """
    Converts image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certain threshold, then takes out
    the first row where a certain percetage of the pixels are above the
    threshold will be the first clip point. Same idea for col, max row, max col.
    """

    percentage = 0.02 # percentage of values greater than a certain threshold

    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0]) # Get a grayscale binary matrix
    # Which is 0.1 above the average, getting an average over the pixel values that are not just black values
    row_sums = np.sum(im, axis=1) # sum the rows
    col_sums = np.sum(im, axis=0) # sum the columns
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(im_crop)

def resize_mantain_aspect(image, desired_size):
    """
    In this function we want to resize but we want to maintain the aspect ratio
    therefore it's going to pad the image with black
    """
    old_size = image.size # old_size[0] is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = image.resize(new_size, Image.ANTIALIAS)
    """
    Cuando ANTIALIAS se agregó inicialmente, era el único filtro de alta calidad basado en circunvoluciones.
    Se suponía que su nombre reflejaría esto. A partir de Pillow 2.7.0, todos los métodos de cambio de tamaño
    se basan en circunvoluciones. Todos ellos son antialias a partir de ahora. Y el nombre real del ANTIALIAS
    filtro es filtro Lanczos.
    """
    new_im = Image.new("RGB", (desired_size, desired_size))
    # New rgb image of the desired size
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def save_single(args):
    img_file, input_path_folder, output_path_folder, output_size = args
    imgname = img_file.split('.')
    image_original = Image.open(os.path.join(input_path_folder, img_file))
    image = trim(image_original)
    image = resize_mantain_aspect(image, desired_size=output_size[0])
    image.save(os.path.join(output_path_folder + imgname[0] + ".jpeg"))

def fast_image_resize(input_path_folder, output_path_folder, output_size=None):
    """
    Uses multiprocessing to make it fast
    """
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        exit()
    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)
    
    jobs = [
        (file, input_path_folder, output_path_folder, output_size)
        for file in os.listdir(input_path_folder)
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))


if __name__ == "__main__":
    fast_image_resize("./Images/DataTest", "./Images/DataTest2/", output_size=(650,650)) 