import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Tuple, List


class Conv1dDropout(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[str, Tuple[int, ...]],
        device=None,
        dtype=None,
    ) -> None:
        super(Conv1dDropout, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self._factory_kwargs = {"device": device, "dtype": dtype}
        self._mask = torch.zeros(
            (out_channels, in_channels, kernel_size), **self._factory_kwargs
        )

    def forward(self, input: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if p < 0 or p > 1:
            raise ValueError("p must be 0~1")
        if training:
            self._mask = self._mask * 0
            random_index = torch.rand((self.kernel_size), **self._factory_kwargs) > p
            self._mask[:, :, random_index] = 1 * (1 / (1 - p + 1e-9))
            masked_weight = self.weight * self._mask
            return F.conv1d(
                input,
                masked_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            return F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )


class TICNN(nn.Module):
    def __init__(self, n_classes: int=10, device: str=None):
        super(TICNN, self).__init__()

        self.dropout_rate = 0.5

        self.first_conv = Conv1dDropout(1, 16, 64, stride=8, padding=28, device=device)

        self.conv_layers = nn.Sequential(
            # conv1
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
            # conv2
            torch.nn.Conv1d(16, 32, 3, stride=1, padding="same"),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
            # conv3
            torch.nn.Conv1d(32, 64, 3, stride=1, padding="same"),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
            # conv4
            torch.nn.Conv1d(64, 64, 3, stride=1, padding="same"),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
            # conv5
            torch.nn.Conv1d(64, 64, 3, stride=1, padding="same"),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
            # conv6
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 2048).to(device)
            dummy = self.first_conv.to(device)(dummy, self.dropout_rate, self.training)
            dummy = self.conv_layers.to(device)(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        self.linear_layers = nn.Sequential(
            torch.nn.Linear(lin_input, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x, self.dropout_rate, self.training)
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)

        return x
