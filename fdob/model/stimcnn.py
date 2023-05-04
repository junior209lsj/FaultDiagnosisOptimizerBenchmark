import torch
from torch import nn
import torch.nn.functional as F

class STIMCNN(nn.Module):
    def __init__(self,
                 in_planes: int=1,
                 n_classes: int=10):
        super(STIMCNN, self).__init__()
        self._conv_layers = nn.Sequential(
            nn.Conv2d(in_planes, 32, 5, 1, "same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1, "same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 28, 28)
            dummy = self._conv_layers(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]
        
        self._linear_layers = nn.Sequential(
            nn.Linear(lin_input, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_layers(x)
        x = torch.flatten(x, 1)
        x = self._linear_layers(x)

        return x