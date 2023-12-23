
from deep_dating.util import dating_util
import torch
import torch.nn as nn
import timm


class DatingCNN(nn.Module):

    IMAGE_NET_MODELS = {"inception_resnet_v2": 299, 
                        "resnet50": 256}

    def __init__(self, model_name="resnet50", pretrained=True, input_size=None):
        super().__init__()

        assert model_name in self.IMAGE_NET_MODELS.keys(), "Unknown model!"
        self.model_name = model_name
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        self.input_size = input_size if input_size else self.IMAGE_NET_MODELS[self.model_name]

        self.optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=0.001)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.base_model(x)

    def save(self, path):
        torch.save(self.state_dict(), path)