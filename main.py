
import dating_util
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dating_metrics import DatingEvalMetricWriter
import matplotlib.pyplot as plt
from dating_datasets import MPS, ScribbleLens, CLaMM, PytorchDatingDataset, SetType
from dating_networks import ResNet50

DATA_PATH = "../datasets/MPS/Download"


mps = MPS()

train_loader = DataLoader(PytorchDatingDataset(mps, SetType.VAL), batch_size=16, shuffle=True, num_workers=8)
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