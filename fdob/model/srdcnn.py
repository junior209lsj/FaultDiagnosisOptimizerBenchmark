import torch
from torch import nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class RDConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(RDConv1d, self).__init__()
        self.padding = padding
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.activation1 = nn.Sigmoid()
        # self.bn1 = nn.BatchNorm1d(num_features=out_channels)

        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.activation2 = nn.Tanh()
        # self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.conv_connection = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=1, stride=2, padding=0)
        
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()

        self.path1 = nn.Sequential(
            self.conv1, self.activation1
        )
        self.path2 = nn.Sequential(
            self.conv2, self.activation2
        )

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        z = out1 * out2
        z = z + self.conv_connection(x)
        z = self.bn(z)
        z = self.relu(z)
        
        return z

class SRDCNN(nn.Module):
    def __init__(self, n_classes: int=10):
        super(SRDCNN, self).__init__()

        self.rd1 = RDConv1d(1, 32, 64, 2, 31, 1)
        self.rd2 = RDConv1d(32, 32, 32, 2, 31, 2)
        self.rd3 = RDConv1d(32, 64, 16, 2, 30, 4)
        self.rd4 = RDConv1d(64, 64, 8, 2, 28, 8)

        self.dense1 = nn.Linear(64*64, 100)
        # self.dense_bn = nn.BatchNorm1d(num_features=100)
        # self.dense_dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(100, n_classes)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.rd1(x)
        x = self.rd2(x)
        x = self.rd3(x)
        x = self.rd4(x)

        x = torch.flatten(x, 1)

        x = self.dense1(x)
        # x = self.dense_bn(x)
        x = self.relu(x)
        # x = self.dense_dropout(x)
        x = self.dense2(x)

        return x