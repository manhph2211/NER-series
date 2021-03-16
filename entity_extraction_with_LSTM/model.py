import torch
import torch.nn as nn
from torch.nn import LSTM

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=True)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=hidden_dim*2, out_features=output_size)
        self.classifier = nn.Softmax()

    def forward(self, X):
        out,hidden = self.lstm(X)
        out = self.relu(out)
        out = out[:, -1, :]
        out = self.linear1(out)
        return out