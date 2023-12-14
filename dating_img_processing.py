
import cv2
from torchvision import transforms

def nn_load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop_size = min(550, min(img.shape))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.RandomCrop(crop_size),
                                    transforms.Resize(256, antialias=False)])
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = transform(img)

    return img