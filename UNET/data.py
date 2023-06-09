
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path): #Clase constructora

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path) # Cuantas imágenes hay

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR) # Leer la imagen a color
        image = image/255.0 ## (512, 512, 3) # Normalizar la imagen 
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512) # para tener el canal primero ya que la capa de entrada que espera el modelo es de 3 canales
        image = image.astype(np.float32) # convertir la imagen en valor flotante de 32
        image = torch.from_numpy(image) # convertir a tensor de un arreglo numpy
        """
            >>> a = numpy.array([1, 2, 3])
            >>> t = torch.from_numpy(a)
            >>> t
            tensor([ 1,  2,  3])
            >>> t[0] = -1
            >>> a
            array([-1,  2,  3])
        """

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
