
import dating_util
import torch
import numpy as np
from tqdm import tqdm
from dating_metrics import DatingMetricWriter
from dating_datasets import MPS, ScribbleLens, CLaMM, DatingDataLoader, SetType
from dating_networks import DatingCNN


model = DatingCNN()
mps = MPS()

train_loader = DatingDataLoader(mps, SetType.VAL, model)
val_loader = DatingDataLoader(mps, SetType.VAL, model)

#train_loader.test_loading()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
num_epochs = 100

metric_writer = DatingMetricWriter()

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
        train_epoch_labels.append(labels.flatten())
        train_epoch_preds.append(outputs.flatten())
        # print(labels.flatten())
        # exit()
        # train_epoch_labels += labels.T.tolist()[0]
        # train_epoch_preds += outputs.T.tolist()[0]
        
        loss.backward()
        model.optimizer.step()
        #model.scheduler.step()

    if epoch % 10:
        model.save(f"model_{epoch}.pt")

    
    metric_writer.epoch(train_epoch_loss, np.concatenate(train_epoch_labels), np.concatenate(train_epoch_preds), epoch)

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