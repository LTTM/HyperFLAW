import torch.nn as nn
import math

from .conditional import Sequential, Conv2d, BatchNorm2d, ReLU6, AdaptiveAvgPool2d, Linear

__all__ = ['mobilenetv2', 'MobileNetV2']


class Sequential(nn.Sequential):
    def forward(self, input, condition: int):
        for module in self:
            input, condition = module(input, condition)
        return input, condition


class Conv2d(nn.Conv2d):
    def forward(self, input, condition):
        input = super().forward(input)
        return input, condition


class BatchNorm2d(nn.Module):
    def __init__(self, 
                 num_features: int,
                 num_data_types: int,
                 eps: float = 0.00001, 
                 momentum: float = 0.1, 
                 affine: bool = True, 
                 track_running_stats: bool = True, 
                 device=None, 
                 dtype=None) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_data_types = num_data_types

        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype) for _ in range(num_data_types)
        ])

    def forward(self, input, condition):
        if not isinstance(condition, int):
            cond = 0
        else:
            cond = condition
        input = self.bns[cond].forward(input)
        return input, condition


class ReLU6(nn.ReLU6):
    def forward(self, input, condition):
        input = super().forward(input)
        return input, condition


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input, condition):
        input = super().forward(input)
        return input, condition


class Linear(nn.Linear):
    def forward(self, input, condition):
        input = super().forward(input)
        return input, condition


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride, num_data_types):
    return Sequential(
        Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup, num_data_types),
        ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, num_data_types):
    return Sequential(
        Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup, num_data_types),
        ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, num_data_types):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = Sequential(
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim, num_data_types),
                ReLU6(inplace=True),
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup, num_data_types),
            )
        else:
            self.conv = Sequential(
                Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim, num_data_types),
                ReLU6(inplace=True),
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim, num_data_types),
                ReLU6(inplace=True),
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup, num_data_types),
            )

    def forward(self, x, condition):
        if self.identity:
            ox, _ = self.conv(x, condition)
            return x + ox, condition
        else:
            return self.conv(x, condition)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., in_channels=3, num_conditions=1):
        super(MobileNetV2, self).__init__()
        self.cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.num_data_types = num_conditions

        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(in_channels, input_channel, 2, num_conditions)]
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, num_conditions))
                input_channel = output_channel

        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        layers.append(conv_1x1_bn(input_channel, output_channel, num_conditions))
        self.features = Sequential(*layers)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.classifier = Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x, condition):
        x, condition = self.features(x, condition)
        x, condition = self.avgpool(x, condition)
        x = x.view(x.size(0), -1)
        x, condition = self.classifier(x, condition)
        return x, condition

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                for bn in m.bns:
                    bn.weight.data.fill_(1)
                    bn.bias.data.zero_()
            elif isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    return MobileNetV2(**kwargs)


if __name__ == "__main__":
    import torch

    model = mobilenetv2(num_data_types=2)
    x = torch.randn(1,3,512,512)

    o, condition = model(x, 0)
    l = o.mean()
    l.backward()

    for n, p in model.named_parameters():
        norm = p.grad.norm() if p.grad is not None else 0
        print(n, norm > 0)