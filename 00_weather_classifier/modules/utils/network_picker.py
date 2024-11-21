from torch.nn import Module
from .. import models as m


def net_picker(name: str) -> Module:
    if name == "conv":
        network = m.ConvolutionalConditionalClassifier
    elif name == "mobile":
        network = m.MobileNetConditionalClassifier
    elif name == "resnet":
        network = m.ResNetConditionalClassifier
    else:
        raise ValueError(f"Invalid model name: {name}")
    
    return network
