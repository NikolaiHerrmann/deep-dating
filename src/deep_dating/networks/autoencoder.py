
import cv2
import torch
import copy
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from pytorch_msssim import ssim
from deep_dating.networks import ModelType
from deep_dating.util import get_torch_device


class Autoencoder(nn.Module):

    def __init__(self, learning_rate=0.001, input_size=512):
        super(Autoencoder, self).__init__()
        self.learning_rate = learning_rate
        self.model_name = "autoencoder"
        self.model_type = ModelType.AUTOENCODER
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            # nn.Linear(512, 65536),
            # nn.ReLU(),
            # nn.Unflatten(1, (256, 16, 16)),
            # nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )



        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() #self.ssim_loss
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        self.metrics = None
        self.classification = False
        self.training_mode = True

    def ssim_loss(self, X, Y):
        return 100 * (1 - ssim(X, Y, data_range=255, size_average=True))

    def forward(self, x):
        if self.training_mode:
            return self.decoder(self.encoder(x))
        return self.encoder(x)
    
    def extract_feature(self, x):
        return self.encoder(x)
    
    def reconstruct_feature(self, x):
        return self.decoder(x)
    
    def transform_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return transforms.ToTensor()(img)

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
        summary(self.encoder, (1, 512, 512))
        summary(self.decoder, (512, 8, 8)) #(256, 32, 32)
        pass
    

if __name__ == "__main__":
    ac = Autoencoder()
    ac.summary()
    