
import dating_util
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import glob
import os
import cv2
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from tqdm import tqdm
from dating_eval_metrics import DatingEvalMetricWriter
import matplotlib.pyplot as plt
import random 
from dating_datasets import MPS, ScribbleLens, CLaMM, PytorchDatingDataset, SetType

DATA_PATH = "../datasets/MPS/Download"


class ResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet.requires_grad_(False)
        resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=1)
        self.base_model = resnet
        self.optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.base_model(x)

    def save(self, path):
        torch.save(self.state_dict(), path)


mps = MPS()

train_loader = DataLoader(PytorchDatingDataset(mps, SetType.TRAIN), batch_size=16, shuffle=True, num_workers=8)
val_loader = DataLoader(PytorchDatingDataset(mps, SetType.VAL), batch_size=16, shuffle=True, num_workers=8)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8)

# images, labels = next(iter(train_loader))
# # helper.imshow(images[0], normalize=False)
# print(labels)
# plt.imshow(images[0][0], cmap="gray")
# plt.show()
# exit()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = ResNet50()

model.to(device)
num_epochs = 100

metric_writer = DatingEvalMetricWriter()

for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    model.train()
    metric_writer.train()

    train_epoch_loss = []
    train_epoch_labels = []
    train_epoch_preds = []
    for inputs, labels in tqdm(train_loader):
        
        inputs = inputs.to(device)
        labels = labels.to(device).reshape((labels.shape[0], 1))
        
        model.optimizer.zero_grad()
        
        outputs = model(inputs)

        loss = model.criterion(outputs, labels)

        train_epoch_loss.append(loss.item())
        train_epoch_labels += labels.T.tolist()[0]
        train_epoch_preds += outputs.T.tolist()[0]
        
        loss.backward()
        model.optimizer.step()
        model.scheduler.step()

    if epoch % 10:
        model.save(f"model_{epoch}.pt")

    metric_writer.epoch(train_epoch_loss, train_epoch_labels, train_epoch_preds, epoch)

    model.eval()
    metric_writer.eval()
    eval_epoch_loss = []
    eval_epoch_labels = []
    eval_epoch_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).reshape((labels.shape[0], 1))
            outputs = model(inputs)

            loss = model.criterion(outputs, labels)
            eval_epoch_loss.append(loss.item())
            eval_epoch_labels += labels.T.tolist()[0]
            eval_epoch_preds += outputs.T.tolist()[0]

    metric_writer.epoch(eval_epoch_loss, eval_epoch_labels, eval_epoch_preds, epoch)