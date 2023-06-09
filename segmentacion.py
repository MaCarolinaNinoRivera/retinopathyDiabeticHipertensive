import os
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from skimage.filters import frangi
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# Lee las imágenes
def readPath(input_path_folder, output_path_folder):
    """
    Uses multiprocessing to make it fast
    """
    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)
    
    jobs = [
        (file, input_path_folder, output_path_folder)
        for file in os.listdir(input_path_folder)
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(saveSingle, jobs), total=len(jobs)))

# Guardar las imagenes en el equipo
def saveSingle(args):
    img_file, input_path_folder, output_path_folder = args  
    imgname = img_file.split('.')
    IMAGE = input_path_folder + img_file 
    image_original = cv2.imread(IMAGE)
    image = makeMasks(image_original) 
    backtorgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    fusion = Image.fromarray(backtorgb)
    fusion.save(os.path.join(output_path_folder + imgname[0] + "_m.jpeg"))

def quitar_ruido(imagen, det, vent, scw):
    det = int(det)
    vent = int(vent)
    scw = int(scw)

    img_new = cv2.fastNlMeansDenoising(imagen, None, det, vent, scw)
    return img_new

# Proceso para tomar la imagen e ir extrayendo las venas
def makeMasks(img):
    # Separación de canales para dejarlas en un solo canal
    blue,green,red = cv2.split(img)
    # Ecualiza el histograma de una imagen en escala de grises utilizando la ecualización de histograma adaptativa limitada de contraste.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11,11)) # Kernel 11
    # Se decide utilizar el canal verde ya que da mejores resultados en más imágenes, esta función aumenta el contraste
    contrast_enhanced_green_fundus = clahe.apply(green)

    # Morfología de apertura y cierre en kernel cada vez más grandes
    R0 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r1 = cv2.morphologyEx(R0, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)	
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    sustraer = cv2.subtract(R2,contrast_enhanced_green_fundus)
    contraste = clahe.apply(sustraer)	

    # Eliminar el ruido 
    sinRuido = quitar_ruido(contraste, 10, 3, 21)

    # Convertir la imagen a una binaria
    _,binaria = cv2.threshold(sinRuido,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Image.fromarray(binaria)

    # imagenoarray = Image.fromarray(binaria)
    # imageGaussian = imagenoarray.filter(ImageFilter.GaussianBlur)

    # imagegausarray = np.array(imageGaussian)

    # imageMedian = cv2.medianBlur(imagegausarray, 11)

    # # Unir las mascaras
    # result1 = cv2.bitwise_and(binaria, imageMedian)

    # # Convertir la imagen a una binaria
    # ret,binaria2 = cv2.threshold(result1, 50,250,cv2.THRESH_BINARY)	
    # Image.fromarray(binaria2)

    return binaria

if __name__ == "__main__":
    readPath('./Images/DataValidation/', './Images/MaskDataValidation/')