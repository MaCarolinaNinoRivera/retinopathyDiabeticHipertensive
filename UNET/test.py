
import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5 # threshold
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1) # se aplana

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    """
        El índice de similitud de Jaccard (a veces denominado coeficiente de similitud de Jaccard )
        compara los miembros de dos conjuntos para ver qué miembros se comparten y cuáles son distintos.
        Es una medida de similitud para los dos conjuntos de datos, con un rango de 0% a 100%.
        Cuanto mayor sea el porcentaje, más similares serán las dos poblaciones
    """
    score_f1 = f1_score(y_true, y_pred)
    """
        La puntuación F1 es una métrica de evaluación de aprendizaje automático que mide la precisión
        de un modelo. Combina las puntuaciones de precisión y recuperación de un modelo.
        f1_score = (TP)/(TP+ 1/2(FP+FN))
    """
    score_recall = recall_score(y_true, y_pred)
    """
        Representa la capacidad del modelo para predecir correctamente los positivos a partir
        de los positivos reales. Esto es diferente a la precisión que mide cuántas predicciones
        hechas por modelos son realmente positivas de todas las predicciones positivas hechas
    """
    score_precision = precision_score(y_true, y_pred)
    """
        La precisión es la relación tp / (tp + fp) donde tp es el número de verdaderos positivos y 
        fp el número de falsos positivos. La precisión es intuitivamente la capacidad del clasificador
        de no etiquetar como positiva una muestra que es negativa. El mejor valor es 1 y el peor valor es 0.
    """
    score_acc = accuracy_score(y_true, y_pred)
    """
        Calcula la puntuación de precisión de un conjunto de etiquetas pronosticadas
        frente a las etiquetas verdaderas. Accuracy Score = (TP+TN)/ (TP+FN+TN+FP)
    """

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42) # Valor de semilla igual que en el train

    """ Folders """
    create_dir("./resultados") # directorio para resultados

    """ Load dataset """
    test_x = sorted(glob("../Data/validationpng/images/*"))
    test_y = sorted(glob("../Data/validationpng/mask/*"))

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "./files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        #print("X: ", x, "; Y: ", y)
        """ Replace \ by / """
        x = x.replace("\\", "/")
        y = y.replace("\\", "/")
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]
        #print("Name: ", name)

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512) lote de una imagen
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        ## mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            """
                Se toma cualquier valor real donde el valor se reduce entre 0 y 1 y el gráfico se reduce a
                la forma de S. También llamada función logística, si el valor de S tiende a infinito positivo,
                entonces la salida se predice como 1 y si el valor llega a infinito negativo, la salida se
                predice como 0. También se denomina clase positiva o clase 1 y clase negativa o clase 0.
            """
            total_time = time.time() - start_time
            time_taken.append(total_time)


            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask) # convertir a 3 canales
        pred_y = mask_parse(pred_y) # convertir a 3 canales
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        cv2.imwrite(f"./resultados/{name}.png", cat_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps) # numero de frames por segundo
