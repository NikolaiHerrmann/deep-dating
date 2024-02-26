
from deep_dating.networks import Autoencoder

class AutoencoderPredictor:

    def __init__(self):
        pass
        
    def run(self, dataset_name, model_path="runs/unet1/v2_model_epoch_168.pt"):
        model = Autoencoder()
        model.load(model_path, continue_training=False)