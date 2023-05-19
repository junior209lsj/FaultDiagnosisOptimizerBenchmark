import torch
from torch import nn
import torch.nn.functional as F


class STFTCNN(nn.Module):
    """
    Implementation of the model by (Zhang et al. 2020),
     an enhanced convolutional neural network for bearing fault diagnosis based on
     time–frequency image (STFT-CNN).

    (Zhang et al. 2020) Ying Zhang, Kangshuo Xing, Ruxue Bai, Dengyun Sun,
     and Zong Meng, “An enhanced convolutional neural network for bearing fault
     diagnosis based on time–frequency image,” Measurement, vol. 157, p. 107667,
     2020, doi: 10.1016/j.measurement.2020.107667.
    """

    def __init__(self, in_planes: int = 1, n_classes: int = 10) -> None:
        """
        Parameters
        ----------
        in_planes: int
            The number of channels of input data.
        n_classes: int
            The number of classes of dataset.
        """
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
            nn.Linear(lin_input, 84), nn.SELU(), nn.Linear(84, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_layers(x)
        x = torch.flatten(x, 1)
        x = self._linear_layers(x)

        return x
