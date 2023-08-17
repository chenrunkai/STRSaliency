import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, input_dim, input_length, dropout: float = 0.25):
        '''
        Default kernal size: 5 and 3, channels: 16 and 16, pooling kernel size: 2 and 2. 
        Linear: 16*(len-8) -> 16 -> 1. 
        Dropout: 0.25. 
        '''
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=5), # input_length-4
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1), # input_length-5
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3), # input_length-7
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1), # input_length-8
        )
        self.dropout = nn.Dropout(dropout)
        self.Linear1 = nn.Linear(16*(input_length-8), 16)
        self.relu = nn.ReLU(inplace=True)
        self.Linear2 = nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        if x.dim()==2:
            x = x.unsqueeze(dim=1)
        x.to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.sigmoid(x)

        return x
    
    
class LSTM(torch.nn.Module):
    def __init__(self, input_dim, output_dim = 1, dropout: float = 0.25, num_layers = 1, bidirectional: bool = False):
        '''
        Default hidden size: 64, num of layers: 1. 
        Linear: 64 -> 16 -> 1. 
        Dropout: defaults to 0.25, but for repetition, 0 is selected for experiments. 
        '''
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=num_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.Linear1 = nn.Linear(64 if self.bidirectional==False else 128, 16)
        self.relu = nn.ReLU(inplace=True)
        self.Linear2 = nn.Linear(16, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        # print("*******************************************************************************")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # input(torch.cuda.is_available())
        # self.device = "cuda"
        self.to(self.device)
        
    def forward(self, x: torch.Tensor):
        # self.device = "cuda:0"
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        if x.dim()==2:
            x = x.unsqueeze(dim=1)
        x.to(self.device)
        x = x.permute(2, 0, 1) # from [batch, input_dim, len] to [len, batch, input_dim]
        h0_dim0 = self.num_layers if self.bidirectional==False else 2*self.num_layers
        # h0 = torch.zeros([h0_dim0, x.shape[1], 64]).to(self.device)
        # c0 = torch.zeros([h0_dim0, x.shape[1], 64]).to(self.device)
        h0 = torch.zeros([h0_dim0, x.shape[1], 64]).to("cuda")
        c0 = torch.zeros([h0_dim0, x.shape[1], 64]).to("cuda")
        # print(self.device)
        x, (h, c) = self.lstm(x, (h0, c0))
        x = self.dropout(x[-1])
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.sigmoid(x)
        return x