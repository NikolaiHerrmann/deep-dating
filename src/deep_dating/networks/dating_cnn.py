
from deep_dating.util import get_torch_device
from deep_dating.networks import ModelType
from deep_dating.metrics import DatingMetrics
import torch
import cv2
import torch.nn as nn

from torchvision import transforms
from torchsummary import summary
import timm


class DatingCNN(nn.Module):

    INCEPTION = "inception_resnet_v2"
    RESNET50 = "resnet50"
    #VGG19 = "vgg19.tv_in1k"
    IMAGE_NET_MODELS = {INCEPTION: 299, RESNET50: 256}
    MODEL_DROP_OUT = {INCEPTION: ("drop", "head_drop"), RESNET50: ("drop_block", None)}

    def __init__(self, model_name, pretrained=True, input_size=None, 
                 learning_rate=0.001, verbose=True, num_classes=None,
                 dropout=True):
        super().__init__()

        assert model_name in self.IMAGE_NET_MODELS.keys(), "Unknown model!"
        self.model_name = model_name
        self.model_type = ModelType.PATCH_CNN
        self.dropout = dropout
        self.verbose = verbose

        if num_classes is None:
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

        if self.dropout:
            drop_conv, drop_head = self.MODEL_DROP_OUT[self.model_name]
            added_dropouts = self.add_dropout(self.base_model, drop_block_name=drop_conv, p_value=0.2, drop_2d=True)
            added_dropouts += self.add_dropout(self.base_model, drop_block_name=drop_head, p_value=0.2, drop_2d=False)
            if self.verbose:
                print(f"Added {added_dropouts} dropout layers")

        self.input_size = input_size if input_size else self.IMAGE_NET_MODELS[self.model_name]
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=learning_rate)
        
        self.transforms = transforms.Compose([
            #transforms.Resize(self.input_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.starting_weights = model_name if pretrained else None
        self.feature_extractor = False

    def add_dropout(self, model, drop_block_name="drop_block", p_value=0.5, drop_2d=True):
        """
        https://discuss.pytorch.org/t/where-and-how-to-add-dropout-in-resnet18/12869/3
        """
        added_dropouts = 0

        if drop_block_name is not None:
            for name, module in model.named_children():

                if len(list(module.children())) > 0:
                    added_dropouts += self.add_dropout(module, drop_block_name=drop_block_name, p_value=p_value, drop_2d=drop_2d)
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
                self._discard_output_layer()
                self.feature_extractor = True
            self.eval()

        if self.verbose:
            print("Model loading completed!")

    def transform_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return self.apply_transforms(img)
    
    def apply_transforms(self, img):
        assert img.shape == (self.input_size, self.input_size), "input has wrong size!"
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return self.transforms(img)
    
    def summary(self):
        #summary(self.base_model, (3, self.input_size, self.input_size))
        pass

if __name__ == "__main__":
    model = DatingCNN(model_name="inception_resnet_v2", input_size=299, num_classes=11)
    model.summary()
