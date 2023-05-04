import torch
from torch import nn
import torch.nn.functional as F

class WDCNNRNN(nn.Module):
    def __init__(self, n_classes: int=10):
        super(WDCNNRNN, self).__init__()

        self.first_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, padding='same')
        self.rnn = nn.LSTM(input_size=16, hidden_size=16, batch_first=True, num_layers=1)
        self.rnn_dropout = nn.Dropout(0.8)

        self.fcnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*128, out_features=100),
            nn.BatchNorm1d(num_features=100),
            nn.Dropout(0.5)
        )

        self.final_layers = nn.Sequential(
            nn.Linear(in_features=100+16, out_features=n_classes)
        )

    def forward(self, x):
        rnn_out = self.first_conv(x)
        rnn_out = torch.permute(rnn_out, (0, 2, 1))
        rnn_out = self.rnn(rnn_out)[1][1][0]
        rnn_out = self.rnn_dropout(rnn_out)

        cnn_out = self.fcnn_layers(x)
        cnn_out = torch.flatten(cnn_out, 1)
        cnn_out = self.linear_layers(cnn_out)

        concat_out = torch.cat((rnn_out, cnn_out), 1)
        concat_out = self.final_layers(concat_out)

        return concat_out