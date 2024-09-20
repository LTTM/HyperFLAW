from collections import OrderedDict

from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torchvision.models.segmentation._utils import _SimpleSegmentationModel

from .conditional import Sequential, Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Dropout


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    def forward(self, x: Tensor, condition: int = None) -> Dict[str, Tensor]:
        # If condition is None, set it to default value
        if condition is None:
            condition = [0 for _ in range(x.shape[0])]

        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features, condition = self.backbone(x, condition)

        result = OrderedDict()
        x = features #["out"]
        result["enc_feats"] = x
        x, feats, condition = self.classifier(x, condition)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x
        result["dec_feats"] = feats
        result["condition"] = condition

        if self.aux_classifier is not None:
            x = features["aux"]
            x, condition = self.aux_classifier(x, condition)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result
    
    def copy_batch(self):
        # Copy all the data of bns[0] in all the other bns
        backbone_state_dict = self.backbone.state_dict()
        bns0 = {}
        for k, v in backbone_state_dict.items():
            try:
                i = k.split(".").index("bns")
                if k.split(".")[i+1] == "0":
                    bns0[k] = v
            except ValueError:
                pass
        
        for k, v in backbone_state_dict.items():
            try:
                i = k.split(".").index("bns")
                splitted_k = k.split(".")
                splitted_k0 = splitted_k[:i+1] + ["0"] + splitted_k[i+2:]
                k0 = ".".join(splitted_k0)
                if splitted_k[i+1] != "0":
                    backbone_state_dict[k] = bns0[k0]
            except ValueError:
                pass
        
        self.backbone.load_state_dict(backbone_state_dict)

class DeepLabHead(Sequential):
    def __init__(self, in_channels: int, num_classes: int, num_data_types=1) -> None:
        super().__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36], num_data_types=num_data_types)
        self.conv1 = Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn = BatchNorm2d(256, num_data_types)
        self.relu = ReLU()
        self.conv2 = Conv2d(256, num_classes, 1)

    def forward(self, x, condition):
        x, condition = self.aspp(x, condition)
        x, condition = self.conv1(x, condition)
        x, condition = self.bn(x, condition)
        feats, condition = self.relu(x, condition)
        x, condition = self.conv2(feats, condition)
        return x, feats, condition

class ASPPConv(Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, num_data_types=1) -> None:
        modules = [
            Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            BatchNorm2d(out_channels,num_data_types),
            ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(Sequential):
    def __init__(self, in_channels: int, out_channels: int, num_data_types=1) -> None:
        super().__init__(
            AdaptiveAvgPool2d(1),
            Conv2d(in_channels, out_channels, 1, bias=False),
            BatchNorm2d(out_channels, num_data_types),
            ReLU(),
        )

    def forward(self, x: torch.Tensor, condition):
        size = x.shape[-2:]
        for mod in self:
            x, condition = mod(x, condition)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False), condition


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256, num_data_types=1) -> None:
        super().__init__()
        modules = []
        modules.append(
            Sequential(Conv2d(in_channels, out_channels, 1, bias=False), BatchNorm2d(out_channels, num_data_types), ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, num_data_types=num_data_types))

        modules.append(ASPPPooling(in_channels, out_channels, num_data_types=num_data_types))

        self.convs = nn.ModuleList(modules)

        self.project = Sequential(
            Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            BatchNorm2d(out_channels, num_data_types),
            ReLU(),
            Dropout(0.5),
        )

    def forward(self, x: torch.Tensor, condition):
        _res = []
        for conv in self.convs:
            _x, condition = conv(x, condition)
            _res.append(_x)
        res = torch.cat(_res, dim=1)
        x, condition = self.project(res, condition)
        return x, condition