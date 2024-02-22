
import cv2
import torch
import copy
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from deep_dating.networks import ModelType
from deep_dating.util import get_torch_device


class Autoencoder(nn.Module):

    def __init__(self, learning_rate=0.001, input_size=256):
        super(Autoencoder, self).__init__()
        self.learning_rate = learning_rate
        self.model_name = "autoencoder"
        self.model_type = ModelType.AUTOENCODER
        self.input_size = input_size

        self.e1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.e2_batch = nn.BatchNorm2d(128)

        self.e3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.e3_batch = nn.BatchNorm2d(256)

        self.e4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.e4_batch = nn.BatchNorm2d(512)

        self.e5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.e5_batch = nn.BatchNorm2d(512)

        self.e6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.e6_batch = nn.BatchNorm2d(512)

        self.e7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.e7_batch = nn.BatchNorm2d(512)

        self.e8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.e8_batch = nn.BatchNorm2d(512)

        #
        self.d1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.d1_batch = nn.BatchNorm2d(512)

        self.d2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.d2_batch = nn.BatchNorm2d(512)

        self.d3 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.d3_batch = nn.BatchNorm2d(512)

        self.d4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.d4_batch = nn.BatchNorm2d(512)

        self.d5 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.d5_batch = nn.BatchNorm2d(256)

        self.d6 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.d6_batch = nn.BatchNorm2d(128)

        self.d7 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.d7_batch = nn.BatchNorm2d(64)

        self.d8 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)

        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.L1Loss()
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        self.metrics = None
        self.classification = False
        self.training_mode = True

        self.transform_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def forward(self, x):
        x1 = self.relu(self.e1(x))
        x2 = self.relu(self.e2_batch(self.e2(x1)))
        x3 = self.relu(self.e3_batch(self.e3(x2)))
        x4 = self.relu(self.e4_batch(self.e4(x3)))
        x5 = self.relu(self.e5_batch(self.e5(x4)))
        x6 = self.relu(self.e6_batch(self.e6(x5)))
        x7 = self.relu(self.e7_batch(self.e7(x6)))
        x8 = self.relu(self.e8_batch(self.e8(x7)))

        xx1 = self.relu(self.d1_batch(self.d1(x8)))
        xx1c = torch.cat([xx1, x7], dim=1)

        xx2 = self.relu(self.d2_batch(self.d2(xx1c)))
        xx2c = torch.cat([xx2, x6], dim=1)

        xx3 = self.relu(self.d3_batch(self.d3(xx2c)))
        xx3c = torch.cat([xx3, x5], dim=1)

        xx4 = self.relu(self.d4_batch(self.d4(xx3c)))
        xx4c = torch.cat([xx4, x4], dim=1)

        xx5 = self.relu(self.d5_batch(self.d5(xx4c)))
        xx5c = torch.cat([xx5, x3], dim=1)

        xx6 = self.relu(self.d6_batch(self.d6(xx5c)))
        xx6c = torch.cat([xx6, x2], dim=1)

        xx7 = self.relu(self.d7_batch(self.d7(xx6c)))
        xx7c = torch.cat([xx7, x1], dim=1)

        xx8 = self.relu(self.d8(xx7c))

        return self.tanh(xx8)
    
    def extract_feature(self, x):
        return self.encoder(x)
    
    def reconstruct_feature(self, x):
        return self.decoder(x)
    
    def transform_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return self.transform_input(img)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def get_state(self):
        return copy.deepcopy(self.state_dict())
    
    def save_state(self, path, state):
        torch.save(state, path)

    def load(self, path, continue_training):
        self.load_state_dict(torch.load(path, map_location=get_torch_device()))

        if continue_training:
            self.starting_weights = path
            self.train()
        else:
            self.training_mode = False
            self.eval()

        #if self.verbose:
        print("Model loading completed!")

    def summary(self):
        # summary(self.encoder, (1, 256, 256))
        # summary(self.decoder, (512, 4, 4)) #(256, 32, 32)
        #summary(self, (1, 256, 256))
        pass
    

if __name__ == "__main__":
    ac = Autoencoder()
    ac.summary()
    