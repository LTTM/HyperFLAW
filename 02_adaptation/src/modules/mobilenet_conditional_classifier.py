import torch
import torch.nn as nn
import torchvision.models as models
from .mobilenetv2 import MobileNetV2
from torchvision._internally_replaced_utils import load_state_dict_from_url

    

class ConditionalClassifier(nn.Module):

    def __init__(self, num_classes=10):
        super(ConditionalClassifier, self).__init__()
        backbone = models.mobilenet_v2(pretrained=True)#MobileNetV2(width_mult=1, in_channels=3)
        self.mobilenetv2 = backbone.features[0] #models.mobilenet_v2(pretrained=True).features[0]
        self.avgpool = nn.AdaptiveAvgPool2d((4, 8))
        self.fc = nn.Linear(1024, num_classes)

        # self.initialize_weights()

    # def initialize_weights(self):
    #     url = 'https://github.com/d-li14/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_1.0-0c6065bc.pth'
    #     state_dict = load_state_dict_from_url(url, progress=True)
    #     state_dict_updated = state_dict.copy()
    #     for k, v in state_dict.items():
    #         if 'features.0.' not in k:
    #             del state_dict_updated[k]
    #         else:
    #             if 'features' not in k and 'classifier' not in k:
    #                 state_dict_updated[k.replace('conv', 'features.18')] = v
    #                 del state_dict_updated[k]
    #
    #     self.mobilenetv2.load_state_dict(state_dict_updated, strict=False)

    def forward(self, x):
        x = self.mobilenetv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
