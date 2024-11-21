import torch
import torch.nn as nn
import torchvision.models as models
from .mobilenetv2 import MobileNetV2
from torchvision._internally_replaced_utils import load_state_dict_from_url


# Define the module exports
__all__ = ['ConditionalClassifier']


class MobileNetConditionalClassifier(nn.Module):

    def __init__(self, num_classes=4):
        super(MobileNetConditionalClassifier, self).__init__()
        self.num_classes = num_classes
        backbone = models.mobilenet_v2(pretrained=True)
        self.mobilenetv2 = backbone.features[0]
        self.avgpool = nn.AdaptiveAvgPool2d((4, 8))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.mobilenetv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_num_classes(self):
        return self.num_classes
