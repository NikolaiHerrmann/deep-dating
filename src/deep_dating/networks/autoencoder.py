
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary


class Autoencoder(nn.Module):

    def __init__(self, learning_rate=0.001):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 128)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 4096),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def extract_feature(self, x):
        return self.encoder(x)
    
    def reconstruct_feature(self, x):
        return self.decoder(x)
    
    def transform_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return transforms.ToTensor()(img)
    

if __name__ == "__main__":
    ac = Autoencoder()
    print(summary(ac.encoder, (1, 512, 512)))
    print(summary(ac.decoder, (128,)))