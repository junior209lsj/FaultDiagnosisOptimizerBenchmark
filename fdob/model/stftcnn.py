import torch
from torch import nn
import torch.nn.functional as F

class STFTCNN(nn.Module):
    def __init__(self,
                 in_planes: int=1,
                 n_classes: int=10) -> None:
        super(STFTCNN, self).__init__()
        self._conv_layers = nn.Sequential(
            nn.Conv2d(in_planes, 6, 5, 1, "same"),
            nn.SELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3, 1, "same"),
            nn.SELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 120, 3, 1, "same"),
            nn.SELU(),
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 64, 64)
            dummy = self._conv_layers(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        self._linear_layers = nn.Sequential(
            nn.Linear(lin_input, 84),
            nn.SELU(),
            nn.Linear(84, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.size())
        x = self._conv_layers(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        x = self._linear_layers(x)
        # print(x.size())

        return x