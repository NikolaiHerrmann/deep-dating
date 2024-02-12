
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from pytorch_msssim import ssim
from deep_dating.networks import ModelType


class Autoencoder(nn.Module):

    def __init__(self, learning_rate=0.001, input_size=512):
        super(Autoencoder, self).__init__()
        self.learning_rate = learning_rate
        self.model_name = "autoencoder"
        self.model_type = ModelType.AUTOENCODER
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Flatten(),
            #nn.Linear(131072, 512)
        )
        
        self.decoder = nn.Sequential(
            # nn.Linear(512, 131072),
            # nn.ReLU(),
            # nn.Unflatten(1, (128, 32, 32)),
            # nn.Upsample(scale_factor=2, mode="nearest"),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode="nearest"),
            # nn.Conv2d(128, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode="nearest"),
            # nn.Conv2d(64, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode="nearest"),
            # nn.Conv2d(32, 1, kernel_size=3, padding=1),
            # nn.Tanh()



            # nn.Upsample(scale_factor=2, mode="nearest"),
            # nn.Conv2d(128, 64, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode="nearest"),
            # nn.Conv2d(64, 32, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # # nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # # nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() #self.ssim_loss
        self.metrics = None
        self.classification = False

    def ssim_loss(self, X, Y):
        return 100 * (1 - ssim(X, Y, data_range=255, size_average=True))

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def extract_feature(self, x):
        return self.encoder(x)
    
    def reconstruct_feature(self, x):
        return self.decoder(x)
    
    def transform_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return transforms.ToTensor()(img)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def summary(self):
        # summary(self.encoder, (1, 512, 512))
        # summary(self.decoder, (512,)) #(256, 32, 32)
        pass
    

if __name__ == "__main__":
    ac = Autoencoder()
    ac.summary()
    