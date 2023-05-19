import torch
from torch import nn
import torch.nn.functional as F


class WDCNN(nn.Module):
    """
    Implementation of the model by (Zhang et al. 2017), Deep Convolutional
     Neural Networks with Wide First-layer Kernels (WDCNN).

    (Zhang et al. 2017) Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen,
     and Zhujun Zhang, “A New Deep Learning Model for Fault Diagnosis with
     Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals,”
     Sensors, vol. 17, no. 2, p. 425, 2017, doi: 10.3390/s17020425.
    """

    def __init__(self, first_kernel: int = 64, n_classes: int = 10) -> None:
        """
        Parameters
        ----------
        first_kernel: int
            The kernel size of the first conv layer.
        n_classes: int
            The number of classes of dataset.
        """
        super(WDCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Conv1
            torch.nn.Conv1d(1, 16, first_kernel, stride=16, padding=24),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            # Pool1
            torch.nn.MaxPool1d(2, 2),
            # Conv2
            torch.nn.Conv1d(16, 32, 3, stride=1, padding="same"),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            # Pool2
            torch.nn.MaxPool1d(2, 2),
            # Conv3
            torch.nn.Conv1d(32, 64, 3, stride=1, padding="same"),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            # Pool3
            torch.nn.MaxPool1d(2, 2),
            # Conv4
            torch.nn.Conv1d(64, 64, 3, stride=1, padding="same"),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            # Pool4
            torch.nn.MaxPool1d(2, 2),
            # Conv5
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            # Pool5
            torch.nn.MaxPool1d(2, 2),
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
            torch.nn.Linear(100, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)

        return x
