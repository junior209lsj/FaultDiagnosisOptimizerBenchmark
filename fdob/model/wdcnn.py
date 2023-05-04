import torch
from torch import nn
import torch.nn.functional as F

class WDCNN(nn.Module):
    def __init__(self, first_kernel: int=64, n_classes: int=10) -> None:
        super(WDCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            #Conv1
            torch.nn.Conv1d(1, 16, first_kernel, stride=16, padding=24),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            #Pool1
            torch.nn.MaxPool1d(2, 2),
            #Conv2
            torch.nn.Conv1d(16, 32, 3, stride=1, padding='same'),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            #Pool2
            torch.nn.MaxPool1d(2, 2),
            #Conv3
            torch.nn.Conv1d(32, 64, 3, stride=1, padding='same'),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            #Pool3
            torch.nn.MaxPool1d(2, 2),
            #Conv4
            torch.nn.Conv1d(64, 64, 3, stride=1, padding='same'),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            #Pool4
            torch.nn.MaxPool1d(2, 2),
            #Conv5
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            #Pool5
            torch.nn.MaxPool1d(2, 2)
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 2048)
            dummy = self.conv_layers(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        self.linear_layers = nn.Sequential(
            torch.nn.Linear(lin_input, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, n_classes)
        )

        # self.reset_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.linear_layers(x)
 
        return x

    # def _init_weight(self, m):
    #     if isinstance(m, nn.Conv1d):
    #         m.reset_parameters()
    #     elif isinstance(m, nn.BatchNorm1d):
    #         m.reset_parameters()
    #     elif isinstance(m, nn.Linear):
    #         m.reset_parameters()

    # def reset_weights(self):
    #     self.apply(self._init_weight)