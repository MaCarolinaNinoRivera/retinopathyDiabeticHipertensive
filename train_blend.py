import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from utils import save_checkpoint, load_checkpoint, check_accuracy
from sklearn.metrics import cohen_kappa_score
import config
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def make_prediction(model, loader, file):
    preds = []
    filenames = []
    model.eval()

    for x, y, files in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            predictions = model(x)
            # Convert MSE floats to integer predictions
            # predictions[predictions < 0.5] = 0
            # predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
            # predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
            # predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
            # predictions[(predictions >= 3.5) & (predictions < 1000000000000)] = 4
            predictions[predictions < 0.5] = 0 # Para el caso de procesamiento MSE MAE
            predictions[(predictions >= 0.5) & (predictions <= 1.5)] = 1 # Para el caso de MSELoss MAE
            predictions[(predictions >= 1.5) & (predictions <= 2.5)] = 2 # Para el caso de MSELoss MAE
            predictions[(predictions >= 2.5) & (predictions <= 3.5)] = 3 # Para el caso de MSELoss MAE
            predictions[(predictions >= 3.5) & (predictions <= 4.5)] = 4 # Para el caso de MSELoss MAE
            predictions[(predictions >= 4.5) & (predictions <= 5.5)] = 5 # Para el caso de MSELoss MAE
            predictions[(predictions >= 5.5) & (predictions <= 6.5)] = 6 # Para el caso de MSELoss MAE
            predictions[(predictions >= 6.5) & (predictions <= 7.5)] = 7 # Para el caso de MSELoss MAE
            predictions[(predictions >= 7.5) & (predictions <= 8.5)] = 8 # Para el caso de MSELoss MAE
            predictions[(predictions >= 8.5) & (predictions <= 100)] = 9 # Para el caso de MSELoss MAE
            predictions = predictions.long().view(-1)
            y = y.view(-1)

            preds.append(predictions.cpu().numpy())
            filenames += map(list, zip(files[0], files[1]))

    filenames = [item for sublist in filenames for item in sublist]
    df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
    df.to_csv(file, index=False)
    model.train()
    print("Done with predictions")

# Leer el archivo CSV creado en la red anterior
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.csv = pd.read_csv(csv_file)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        example = self.csv.iloc[index, :]
        features = example.iloc[: example.shape[0] - 4].to_numpy().astype(np.float32)
        labels = example.iloc[-4:-2].to_numpy().astype(np.int64)
        filenames = example.iloc[-2:].values.tolist()
        return features, labels, filenames


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d((1280 + 1) * 2), # Aplica la normalización por lotes sobre una entrada 2D o 3D
            nn.Linear((1280+1) * 2, 500), # nn linear es un módulo que se utiliza para crear una red de alimentación directa de una sola capa con n entradas y m salidas.
            nn.BatchNorm1d(500),
            nn.ReLU(), # (Rectified Linear Unit) función de activación
            nn.Dropout(0.2), # Establece aleatoriamente los elementos en cero para evitar el sobreajuste.
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = MyModel().to(config.DEVICE)
    ds = MyDataset(csv_file="./train_blend.csv")
    loader = DataLoader(ds, batch_size=256, num_workers=3, pin_memory=True, shuffle=True)
    ds_val = MyDataset(csv_file="./train_blend.csv")
    loader_val = DataLoader(
        ds_val, batch_size=256, num_workers=3, pin_memory=True, shuffle=True
    )
    ds_test = MyDataset(csv_file="./test_blend.csv")
    loader_test = DataLoader(
        ds_test, batch_size=256, num_workers=2, pin_memory=True, shuffle=False
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    

    if config.LOAD_MODEL and "linear.pth.tar" in os.listdir():
        load_checkpoint(torch.load("linear.pth.tar"), model, optimizer, lr=1e-4)
        model.train()

    for i in range(5):
        print("\n Epocha número ", i)
        losses = []
        for x, y, files in tqdm(loader_val):
            x = x.to(config.DEVICE).float()
            y = y.to(config.DEVICE).view(-1).float()

            # forward
            scores = model(x).view(-1)
            loss = loss_fn(scores, y)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        print(f"\nLoss: {sum(losses)/len(losses)}")

    if config.SAVE_MODEL:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename="linear.pth.tar")

    target_names = ['0','1','2','3','4','5','6','7','8','9']
    
    print("\n Check_accuracy Validación")
    preds, labels = check_accuracy(loader_val, model)
    print("\n Loader_val", cohen_kappa_score(labels, preds, weights="quadratic"))

    # Accuracy
    from sklearn.metrics import accuracy_score
    print("\n-------------------------------Accuracy Validación")
    print(accuracy_score(labels, preds))
    # Recall
    from sklearn.metrics import recall_score
    print("\n-------------------------------Recall Validación")
    print(recall_score(labels, preds, average=None))
    # f1
    print("\n-------------------------------f1 Validación")
    print(f1_score(labels, preds, average=None))
    # Method 3: Classification report
    print("\n-------------------------------Classification report Validación")
    print(classification_report(labels, preds, target_names=target_names))

    # Confusion Matrix
    print("\n---------------------------Matriz de confusión Validación")
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics
    print(confusion_matrix(labels, preds))
    matrix = confusion_matrix(labels, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = target_names)
    cm_display.plot()
    # plt.show()
    plt.savefig("MatrizConfusionValidation.jpg")

    print("\n Check Accuracy train")
    preds, labels = check_accuracy(loader, model)
    print("\loader", cohen_kappa_score(labels, preds, weights="quadratic"))

    # Accuracy
    from sklearn.metrics import accuracy_score
    print("\n-------------------------------Accuracy Train")
    print(accuracy_score(labels, preds))
    # Recall
    from sklearn.metrics import recall_score
    print("\n-------------------------------Recall Train ")
    print(recall_score(labels, preds, average=None))
    # F1
    print("\n-------------------------------f1 Train ")
    print(f1_score(labels, preds, average=None))
    # Method 3: Classification report
    print("\n-------------------------------Classification report Validación")
    print(classification_report(labels, preds, target_names=target_names))
    # Confusion Matrix
    print("\n---------------------------Matriz de confusión Validación")
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics
    print(confusion_matrix(labels, preds))
    matrix = confusion_matrix(labels, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = target_names)
    cm_display.plot()
    # plt.show()
    plt.savefig("MatrizConfusionTraining.jpg")

    print("\n ++++++++++++++++++ Comienza Make Prediction Test")
    make_prediction(model, loader_test, "test_preds.csv")
