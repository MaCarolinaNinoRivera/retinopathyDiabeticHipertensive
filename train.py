# Donde se entrena el modelo
from statistics import mode
import torch
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset
from torchvision.utils import save_image
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    make_prediction,
    get_csv_for_blend # para el caso de ajuste pesos
)
import pandas as pd #For reading csv files.
import matplotlib.pyplot as plt #For plotting.
import numpy as np 
import pickle

# train_ds = DRDataset(
#         images_folder="./DataTrain/",
#         path_to_csv="./Train.csv",
#         transform=config.train_transforms
#     )
# train_loader = DataLoader(
#         train_ds,
#         batch_size=config.BATCH_SIZE,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY,
#         shuffle=True,
#     )
# valid_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, _) in enumerate(loop):

        # save examples and make sure they look ok with the data augmentation,
        # tip is to first set mean=[0,0,0], std=[1,1,1] so they look "normal"
        # save_image(data, f"hi_{batch_idx}.png")

        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            # loss = loss_fn(scores, targets) # Cross entropy 
            loss = loss_fn(scores, targets.unsqueeze(1).float()) # MSE Loss MAE

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    print(f"\n++++++++++++++++ Loss average over epoch: {sum(losses)/len(losses)}")
    lossestotal = sum(losses)/len(losses)
    return lossestotal


def main():
    train_ds = DRDataset(
        # images_folder="G:/Mi unidad/Colab Notebooks/retinopathy/DiaHiper/Total650/Train650/",
        # images_folder="C:/Users/carol/OneDrive/Documentos/Sergio/KaggleRetinopathy/baseLine/Data/Train/resized_650/",
        # path_to_csv="./traintotal.csv",
        # path_to_csv="./trainLabels.csv",
        images_folder="./Images/DataTrain/",
        path_to_csv="./Train.csv",
        transform=config.train_transforms,
        # transform=config.val_transforms, # Proceso blending
    )
    val_ds = DRDataset(
        # images_folder="G:/Mi unidad/Colab Notebooks/retinopathy/DiaHiper/Total650/Validation650/",
        # images_folder="C:/Users/carol/OneDrive/Documentos/Sergio/KaggleRetinopathy/baseLine/Data/Train/resized_650/",
        # path_to_csv="./validationtotal.csv",
        # path_to_csv="./trainLabels.csv",
        images_folder="./Images/DataValidation/",
        path_to_csv="./validation.csv",
        transform=config.val_transforms,
    )
    test_ds = DRDataset(
        # images_folder="G:/Mi unidad/Colab Notebooks/retinopathy/DiaHiper/Total650/Test650/",
        # images_folder="C:/Users/carol/OneDrive/Documentos/Sergio/KaggleRetinopathy/baseLine/Data/Test/resized_650/",
        # path_to_csv="./testTotal.csv",
        images_folder="./Images/DataTest2/",
        path_to_csv="./Test.csv",
        transform=config.val_transforms,
        train=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        # shuffle=False, # Proceso blending
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    #Histogram of label counts.
    # path = "./"

    # train_df = pd.read_csv(f"{path}trainLabels.csv")
    # print(f'No.of.training_samples: {len(train_df)}')
    # print(train_df)
    # print("df:", train_df)
    # train_df.level.hist()
    # plt.xticks([0,1,2,3,4])
    # plt.grid(False)
    # plt.show() 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Use GPU if it's available or else use CPU.
    # #As you can see,the data is imbalanced.
    # #So we've to calculate weights for each class,which can be used in calculating loss.
    # from sklearn.utils import class_weight #For calculating weights for each class.
    # class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.array([0,1,2,3,4]),y=train_df['level'].values)
    # class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    
    # loss_fn =  nn.CrossEntropyLoss(weight=class_weights)
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.L1Loss() # MAE
    loss_fn = nn.MSELoss()
    model = EfficientNet.from_pretrained("efficientnet-b0")
    # model._fc = nn.Linear(1536, 1) # Número de clases a realizar
    # model._dropout = nn.Dropout(0.1)
    model._fc = nn.Linear(1280, 1) # En vez de usar 5 predicciones diferentes solo se regresará uno en el cambio loss
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

    # Run after training is done and you have achieved good result
    # on validation set, then run train_blend.py file to use information
    # about both eyes concatenated
    # get_csv_for_blend(val_loader, model, "./val_blend.csv")
    # get_csv_for_blend(test_loader, model, "./test_blend.csv")
    # get_csv_for_blend(train_loader, model, "./train_blend.csv")
    # make_prediction(model, test_loader) 
    # import sys
    # sys.exit()

    # model._fc = nn.Linear(1536,1)

    # Prueba de uno solo
    # make_prediction(model, test_loader)
    # import sys
    # sys.exit()
    best_valid_loss = float("inf")
    checkpoint_path = "./b0.pth.tar"
    for epoch in range(config.NUM_EPOCHS):
        print("\n*********************************** Comienza la época ", epoch, " *************************")
        valid_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        # get on validation
        print("\n ---------------------------------- Comienza chequear accuracy ----------------------------")
        preds, labels = check_accuracy(val_loader, model, config.DEVICE)
        print("tipos ", type(preds), type(labels))
        print("\n ***///***///Preds", preds)
        print("\n ***///***///Labels", labels)
        print(f"\n---------------------- QuadraticWeightedKappa (Validation): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        print("\n preds ", preds, "labels, ", labels)

        # Confusion Matrix
        print("\n---------------------------Matriz de confusión")
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(labels, preds))
        matrix = confusion_matrix(labels, preds)
        # Accuracy
        from sklearn.metrics import accuracy_score
        print("\n-------------------------------Accuracy")
        print(accuracy_score(labels, preds))
        # Recall
        from sklearn.metrics import recall_score
        print("\n-------------------------------Recall")
        print(recall_score(labels, preds, average=None))


        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            # save_checkpoint(checkpoint, filename=f"b3_{epoch}.pth.tar") # archivo por epocas
            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
                print(data_str)
                best_valid_loss = valid_loss
                save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE) # archivo total
    print("\n ++++++++++++++++++ Comienza Make Prediction ")
    make_prediction(model, test_loader)   
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)

if __name__ == "__main__":
    main()