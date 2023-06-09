import torch
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
import warnings
import torch.nn.functional as F


def make_prediction(model, loader, output_csv="submission.csv"):
    # print("\n Make prediction ")
    preds = []
    filenames = []
    model.eval()

    for x, y, files in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            # print("Files ", files)
            # predictions = model(x).argmax(1) # Para el caso de pre procesamiento con nn.CrossEntropyLoss()
            # preds.append(predictions.cpu().numpy) # Para el caso de procesamiento nn.CrossEntropyLoss()
            predictions = model(x) # para el caso de MSE o MAE
            # print("\n***/**/*/*/*/*/*/* Make predictions", predictions)
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
            predictions = predictions.long().squeeze(1) # Para el caso de MSELoss MAE
            preds.append(predictions.cpu().numpy()) # Para el caso deMSELoss MAE
            filenames += files 
    # print("\n Valor de pred", predictions, "preds", preds)
    df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
    df.to_csv(output_csv, index=False)
    model.train()
    print("\n /////////////// ************** Done with predictions ************** ////////////////")


def check_accuracy(loader, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    # print("\nCheckAccuracy")
    model.eval()
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0

    for x, y, filename in tqdm(loader):
        # print("Filename", filename)
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            # scores = model(x) # para el caso de pre proceso nn.CrossEntropyLoss()
            predictions = model(x) # Para el caso de procesamiento MSELoss MAE
        # _, predictions = scores.max(1) # para el caso de pre proceso nn.CrossEntropyLoss()
        # num_correct += (predictions == y).sum() # para el caso de pre proceso nn.CrossEntropyLoss()
        # num_samples += predictions.shape[0] # para el caso de pre proceso nn.CrossEntropyLoss()
        # Convert MSE MAE floats to integer predictions
        # print("\n***/**/*/*/*/*/*/* Check predictions", predictions)
        # print("y¿¿¿¿¿¿¿¿¿¿¿¿!!!!!!!!!11", y)
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
        predictions = predictions.long().view(-1) # Para el caso de procesamiento MSE MAE
        y = y.view(-1) # Para el caso de procesamiento MSE MAE
        # print("\n***/**/*/*/*/*/*/* Check predictions--------------------", predictions)
        num_correct += (predictions == y).sum() 
        num_samples += predictions.shape[0] 

        # add to lists
        all_preds.append(predictions.detach().cpu().numpy()) 
        all_labels.append(y.detach().cpu().numpy()) 
        
    print(
        f"\n------------ Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )
    model.train()
    return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(
        all_labels, axis=0, dtype=np.int64
    )


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("\n => Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("\n =======> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# Paso importante para generar la segunda red, genera un csv con todas las características
# del modelo, ojo derecho e izquierdo
def get_csv_for_blend(loader, model, output_csv_file):
    # warnings.warn("Important to have shuffle=False (and to ensure batch size is even size) when running get_csv_for_blend also set val_transforms to train_loader!")
    model.eval()
    filename_first = []
    filename_second = []
    labels_first = []
    labels_second = []
    all_features = []

    for idx, (images, y, image_files) in enumerate(tqdm(loader)): # iteración de imágenes
        images = images.to(config.DEVICE)
        
        with torch.no_grad():
            features = F.adaptive_avg_pool2d(
                model.extract_features(images), output_size=1
            )
            """
                En la extracción de características, comenzamos con un modelo previamente entrenado y 
                solo actualizamos los pesos de capa finales de los que derivamos las predicciones. 
                Se llama extracción de características porque usamos la CNN preentrenada como un extractor
                de características fijo y solo cambiamos la capa de salida.
            """
            if features.shape[0] % 2 == 0:
                # Dividir el lote en dos correspondiente a ojo derecho e izquierdo 
                features_logits = features.reshape(features.shape[0] // 2, 2, features.shape[1])
                preds = model(images).reshape(images.shape[0] // 2, 2, 1)
                # Obtener la predicción del modelo y concatenar los dos ya que la predicción
                # es solo un valor escalar y se aplana con la vista y se hace cpu y numpy
                new_features = (
                    torch.cat([features_logits, preds], dim=2)
                    .view(preds.shape[0], -1)
                    .cpu()
                    .numpy()
                )
                all_features.append(new_features)
                filename_first += image_files[::2]
                filename_second += image_files[1::2]
                labels_first.append(y[::2].cpu().numpy())
                labels_second.append(y[1::2].cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    df = pd.DataFrame(
        data=all_features, columns=[f"f_{idx}" for idx in range(all_features.shape[1])]
    )
    df["label_left"] = np.concatenate(labels_first, axis=0)
    df["label_right"] = np.concatenate(labels_second, axis=0)
    df["file_left"] = filename_first
    df["file_right"] = filename_second
    df.to_csv(output_csv_file, index=False)
    model.train()
