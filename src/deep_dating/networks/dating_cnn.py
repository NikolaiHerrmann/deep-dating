
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torchinfo import summary
from PIL import Image
from deep_dating.util import get_torch_device
from deep_dating.networks import ModelType
from deep_dating.metrics import DatingMetrics


class DatingCNN(nn.Module):

    RESNET50 = "resnet50"
    INCEPTION = "inception_resnet_v2"
    EFFICIENTNET_B4 = "tf_efficientnet_b4"

    IMAGE_NET_MODELS = {INCEPTION: 299, RESNET50: 256, EFFICIENTNET_B4: 380}

    # For pipeline 2 do: INCEPTION: [("drop", 0.1, True), ("head_drop", 0.2, False)]

    MODEL_DROP_OUT = {INCEPTION: [("drop", 0.1, True), ("head_drop", 0.2, False)], 
                      RESNET50: [("drop_block", 0.2, True)],
                      EFFICIENTNET_B4: [("drop", 0.1, True)]}

    def __init__(self, model_name, pretrained=True, input_size=None, 
                 learning_rate=0.001, verbose=True, num_classes=None,
                 dropout=True, resize=False):
        super().__init__()

        assert model_name in self.IMAGE_NET_MODELS.keys(), "Unknown model!"
        self.model_name = model_name
        self.model_type = ModelType.PATCH_CNN
        self.dropout = dropout
        self.resize = resize
        self.verbose = verbose

        if num_classes is None or num_classes == 1:
            num_classes = 1
            self.classification = False
            self.criterion = nn.MSELoss()
            self.final_activation = nn.Identity()
        else:
            self.classification = True
            self.criterion = nn.NLLLoss()
            self.final_activation = nn.LogSoftmax(dim=1) 
        
        self.metrics = DatingMetrics(classification=self.classification)

        if self.verbose:
            print("Model doing:", ("classification" if self.classification else "regression"))

        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        self.dropout_vals = None
        if self.dropout:
            num_dropouts = 0
            self.dropout_vals = self.MODEL_DROP_OUT[self.model_name]
            for block_name, p_value, is_2d in self.dropout_vals:
                num_dropouts += self.add_dropout(self.base_model, drop_block_name=block_name, p_value=p_value, drop_2d=is_2d)
            if self.verbose:
                print(f"Added {num_dropouts} dropout layers")

        self.input_size = input_size if input_size else self.IMAGE_NET_MODELS[self.model_name]
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
        
        transform_array = [
            transforms.Resize(self.input_size, antialias=True, max_size=None, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if not self.resize:
            transform_array.pop(0)
        self.transform_input = transforms.Compose(transform_array)

        self.starting_weights = model_name if pretrained else None
        self.feature_extractor = False

    def add_dropout(self, model, drop_block_name, p_value, drop_2d):
        """
        https://discuss.pytorch.org/t/where-and-how-to-add-dropout-in-resnet18/12869/3
        """
        added_dropouts = 0

        if drop_block_name is not None:
            for name, module in model.named_children():

                if len(list(module.children())) > 0:
                    added_dropouts += self.add_dropout(module, drop_block_name, p_value, drop_2d)
                if name == drop_block_name:
                    dropout = nn.Dropout2d(p=p_value) if drop_2d else nn.Dropout(p=p_value)
                    setattr(model, name, dropout)
                    added_dropouts += 1

        return added_dropouts

    def forward(self, x):
        return self.final_activation(self.base_model(x))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def _discard_output_layer(self):        
        discard_layer = nn.Identity()
        classifier_layer = self.base_model.pretrained_cfg["classifier"]
        setattr(self.base_model, classifier_layer, discard_layer)

        assert type(self.base_model.get_classifier()) == type(discard_layer), "Failed to change classifier!"

        if self.verbose:
            print("Changed last layer to:", type(discard_layer))

    def load(self, path, continue_training, use_as_feat_extractor=False):
        self.load_state_dict(torch.load(path, map_location=get_torch_device(self.verbose)))

        if continue_training:
            self.starting_weights = path
            self.train()
        else:
            if use_as_feat_extractor:
                self.use_as_feature_extractor()
            self.eval()

        if self.verbose:
            print("Model loading completed!")

    def use_as_feature_extractor(self):
        self._discard_output_layer()
        self.feature_extractor = True
        self.eval()

    def transform_img(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self.apply_transforms(img)
    
    def apply_transforms(self, img):
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return self.transform_input(img)
    
    def summary(self, batch_size=32):
        summary(self.base_model, input_size=(batch_size, 3, self.input_size, self.input_size), device=get_torch_device(self.verbose))


if __name__ == "__main__":
    model = DatingCNN(model_name=DatingCNN.EFFICIENTNET_B5, num_classes=15)
    print(model.base_model)
    model.summary()
