
import dating_util
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

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