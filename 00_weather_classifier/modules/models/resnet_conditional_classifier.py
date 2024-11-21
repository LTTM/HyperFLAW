import torch.nn as nn
import torchvision.models as models


# Define the module exports
__all__ = ['ResNetConditionalClassifier']


class ResNetConditionalClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetConditionalClassifier, self).__init__()

        self.num_classes = num_classes
        
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
    def get_num_classes(self):
        return self.num_classes
