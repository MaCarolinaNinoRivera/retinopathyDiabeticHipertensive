
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
from tqdm import tqdm

def train(model, loader, optimizer, loss_fn, device):
    print("Start Train: ")
    epoch_loss = 0.0

    model.train()
    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad() # Establece los gradientes de todos torch.Tensorlos s optimizados en cero.
        """
            En PyTorch, para cada mini lote durante la fase de entrenamiento , normalmente queremos
            establecer explícitamente los gradientes en cero antes de comenzar a hacer la retropropagación
            (es decir, actualizar los pesos y sesgos ) porque PyTorch acumula los gradientes en los pases
            hacia atrás posteriores. Este comportamiento de acumulación es conveniente cuando se entrenan
            RNN o cuando queremos calcular el gradiente de la pérdida sumada en varios minilotes.
            Por lo tanto, la acción predeterminada se ha configurado para acumular (es decir, sumar) los
            gradientes en cada loss.backward() llamada.
        """
        y_pred = model(x) # hacer la predicción del modelo
        loss = loss_fn(y_pred, y)
        loss.backward() 
        """
            loss.backward()calcula dloss/dxpara cada parámetro x que tiene requires_grad=True.
            Estos se acumulan en x.gradpara cada parámetro x.
        """
        optimizer.step() # Realiza un único paso de optimización (actualización de parámetros).
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    print("In evaluate")
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42) # valor semilla

    """ Directories """
    create_dir("./files") # guardar el checkpoint

    """ Load dataset """
    train_x = sorted(glob("../Data/trainpng/images/*"))# traer todo con el *
    train_y = sorted(glob("../Data/trainpng/mask/*"))

    valid_x = sorted(glob("../Data/testpng/images/*"))
    valid_y = sorted(glob("../Data/testpng/mask/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n" # mostrar el tamaño del aumento
    print(data_str)

    """ Hyperparameters """
    H = 650
    W = 650
    size = (H, W)
    batch_size = 2
    num_epochs = 10
    lr = 1e-4
    checkpoint_path = "./files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cpu')   ## GTX 1060 6GB
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    """
        Reduzca la tasa de aprendizaje cuando una métrica ha dejado de mejorar. Los modelos a menudo se 
        benefician al reducir la tasa de aprendizaje en un factor de 2 a 10 una vez que el aprendizaje se 
        estanca. Este planificador lee una cantidad de métricas y, si no se observa ninguna mejora durante 
        un número de épocas de "paciencia", la tasa de aprendizaje se reduce.
    """
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        print("Epoca", epoch)
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
