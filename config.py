# Especificar todos los hiperparámetros del modelo
import torch
#	una biblioteca Tensor como NumPy, con fuerte soporte GPU
# Por lo general, PyTorch se usa como:

# Un reemplazo para NumPy para usar el poder de las GPU.
# Una plataforma de investigación de aprendizaje profundo que proporciona la máxima flexibilidad y velocidad.
import albumentations as A
# Albumentations es una herramienta de visión artificial que aumenta 
# el rendimiento de las redes neuronales convolucionales profundas.
from albumentations.pytorch import ToTensorV2 
#Es la manera en la que Albumentations transforma en tensores de Pytorch

# Especificación de los hiperparametros
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4 # Determina el tamaño del paso en cada iteración mientras avanza hacia un mínimo de una función de pérdida
WEIGHT_DECAY = 5e-4 # Técnica de regularización que agrega una pequeña penalización a una función de pérdida, se utiliza para evitar el sobreajuste
BATCH_SIZE = 64 # Hiperparámetro que define el número de muestras para trabajar antes de actualizar los parámetros del modelo interno.
NUM_EPOCHS = 20 # Hiperparámetro que define el número de veces que el algoritmo de aprendizaje funcionará en todo el conjunto de datos de entrenamiento.
NUM_WORKERS = 4 # Le dice a la instancia del cargador de datos cuántos subprocesos usar para la carga de datos. Si num_worker es cero (predeterminado), la GPU tiene que pesar para que la CPU cargue datos. Teóricamente, cuanto mayor sea el num_workers, más eficientemente la CPU cargará datos y menos la GPU tendrá que esperar.
CHECKPOINT_FILE = "b0.pth.tar" # Construir un objeto para guardar uno solo o un grupo de objetos rastreables en un archivo de punto de control. Mantiene un save_counterpunto de control para la numeración.
# Donde conoce el archivo en el que se guarda el modelo
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

# Aumento de datos para imágenes
train_transforms = A.Compose(
    [
        A.Resize(width=130, height=130),
        # A.RandomCrop(height=120, width=120), #Recorte aleatorio
        # # Heavy aumentation
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        #A.Blur(p=0.3),
        A.CLAHE(p=0.3), # Contrast-limited adaptive histogram equalization implemented in tensorflow ops.
        # A.ColorJitter(p=0.3),
        # A.CoarseDropout(max_holes=6, max_height=10, max_width=10, p=0.3),
        # A.Affine(p=0.2),
        # A.AdvancedBlur(p=0.3),
        # A.Defocus(p=0.5),
        # A.Sharpen(p=0.5),
        # A.FancyPCA(p=0.3),
        # End heavy aumentation
        A.Normalize(
            mean=[0.4209, 0.2808, 0.1770], # Media
            std=[0.2585, 0.1778, 0.1162], # Desviación estándar
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=120, width=120),
        A.Normalize(
            mean=[0.4209, 0.2808, 0.1770], # Media
            std=[0.2585, 0.1778, 0.1162], # Desviación estándar
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)