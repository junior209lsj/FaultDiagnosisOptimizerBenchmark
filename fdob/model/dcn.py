import torch
from torch import nn
import torch.nn.functional as F


class DCN(nn.Module):
    def __init__(
        self, in_channels: int = 1, n_classes: int = 10, residual_coef: float = 0.2
    ):
        super(DCN, self).__init__()

        self.residual_block = ResidualConnectionBlock(
            channel=in_channels, residual_coef=residual_coef
        )
        self.dilated_block = DilatedResidualConnectionBlock(
            channel=32, residual_coef=residual_coef
        )

        self.se1 = SEBlock(channel=32)
        self.se2 = SEBlock(channel=64)

        self.global_connection_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, stride=32)
        )

        self.last_pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 13, out_features=100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, n_classes),
        )

        self.residual_coef = residual_coef

    def forward(self, x):
        out = self.residual_block(x)
        out = self.se1(out)
        out = self.dilated_block(out)
        out = self.se2(out)

        out = out + (self.global_connection_layers(x) * self.residual_coef)

        out = self.last_pool(out)

        out = torch.flatten(out, 1)
        out = self.linear_layers(out)

        return out


class ResidualConnectionBlock(nn.Module):
    def __init__(self, channel: int = 1, residual_coef: float = 0.2):
        super(ResidualConnectionBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=channel, out_channels=16, kernel_size=64, stride=8, padding=28
        )
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3
        )

        self.conv_connection = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=1, stride=16
        )

        self.bn_out = nn.BatchNorm1d(num_features=32)
        self.pool_out = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)

        self.relu = nn.ReLU()

        self.conv_layers = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.pool1, self.conv2
        )

        self.connection_layers = nn.Sequential(self.conv_connection)

        self.output_layers = nn.Sequential(self.bn_out, self.relu, self.pool_out)

        self.residual_coef = residual_coef

    def forward(self, x):
        out = self.conv_layers(x) + (self.connection_layers(x) * self.residual_coef)
        out = self.output_layers(out)
        return out


class DilatedResidualConnectionBlock(nn.Module):
    def __init__(self, channel: int = 32, residual_coef: float = 0.2):
        super(DilatedResidualConnectionBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=channel,
            out_channels=64,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding="same",
        )
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            dilation=2,
            padding="same",
        )
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            dilation=3,
            padding="same",
        )

        self.conv_connection = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=1, stride=1, padding="same"
        )

        self.bn_out = nn.BatchNorm1d(num_features=64)

        self.relu = nn.ReLU()

        self.conv_layers = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu, self.conv3
        )

        self.connection_layers = nn.Sequential(self.conv_connection)

        self.output_layers = nn.Sequential(self.bn_out, self.relu)

        self.residual_coef = residual_coef

    def forward(self, x):
        out = self.conv_layers(x) + (self.connection_layers(x) * self.residual_coef)
        out = self.output_layers(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc_layers(y).view(b, c, 1)

        return x * y.expand_as(x)
