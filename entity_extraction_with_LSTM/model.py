import torch
import torch.nn as nn
from torch.nn import LSTM
import config



class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTM, self).__init__()
        self.em = nn.Embedding(N_WORDS,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(in_features=hidden_dim*2, out_features=output_size)
        

    def forward(self, X):
        X = self.em(X)
        out,hidden = self.lstm(X)
        X = X.permute(0,2,1)
        out = self.relu(out)
        #out = out[:,-1,:]
        out = self.drop(out)
        out = self.linear1(out)
        return out

#	model = LSTM(input_size=MAX_LEN, output_size=N_CLASSES, hidden_dim=64, n_layers=2)
