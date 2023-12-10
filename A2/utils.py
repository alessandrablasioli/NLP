# pytorch
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

# pytorch lightning
from lightning import LightningModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics.classification import MultilabelF1Score



def model_predict_3(model, dataloader):
    model.eval()  

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch

            X_1, X_2, X_3 = data["Premise"], data["Conclusion"], data["Conclusion"]

            encoded_1 = model.tokenizer(X_1, padding=True, truncation=True, return_tensors="pt")
            encoded_2 = model.tokenizer(X_2, padding=True, truncation=True, return_tensors="pt")
            encoded_3 = model.tokenizer(X_3, padding=True, truncation=True, return_tensors="pt")

            batch_predictions = model(encoded_1, encoded_2, encoded_3)
            predictions.append(batch_predictions)

    all_predictions = torch.cat(predictions)

    return all_predictions



def model_predict_2(model, dataloader):
    model.eval()  

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch

            X_1, X_2 = data["Premise"], data["Conclusion"]

            encoded_1 = model.tokenizer(X_1, padding=True, truncation=True, return_tensors="pt")
            encoded_2 = model.tokenizer(X_2, padding=True, truncation=True, return_tensors="pt")

            batch_predictions = model(encoded_1, encoded_2)
            predictions.append(batch_predictions)

    all_predictions = torch.cat(predictions)

    return all_predictions


def model_predict_1(model, dataloader):
    model.eval()  

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch

            X_1 = data["Premise"]

            encoded_1 = model.tokenizer(X_1, padding=True, truncation=True, return_tensors="pt")

            batch_predictions = model(encoded_1)
            predictions.append(batch_predictions)

    all_predictions = torch.cat(predictions)

    return all_predictions


