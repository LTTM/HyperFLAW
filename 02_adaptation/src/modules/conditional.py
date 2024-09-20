import torch.nn as nn
class Sequential(nn.Sequential):
    def forward(self, input, condition):
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
            assert all(con == condition[0] for con in condition)
            cond = condition[0]
        else:
            cond = condition
        input = self.bns[cond].forward(input)
        return input, condition


class ReLU6(nn.ReLU6):
    def forward(self, input, condition):
        input = super().forward(input)
        return input, condition

class ReLU(nn.ReLU):
    def forward(self, input, condition):
        input = super().forward(input)
        return input, condition

class Dropout(nn.Dropout):
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
