import torch.nn as nn
import torch

class FNN(nn.Module): #feedforward neural network with 3 hidden layers
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim=1):
        super(FNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.sig1 = nn.Sigmoid()

        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim) 
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.sig2 = nn.Sigmoid()

        self.fc3 = nn.Linear(hidden2_dim, output_dim) 
        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('linear'))

        # self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        # nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('sigmoid'))
        # self.sig3 = nn.Sigmoid()

        # self.fc4 = nn.Linear(hidden3_dim, output_dim)
        # nn.init.xavier_uniform_(self.fc4.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        x = self.fc1(x)
        x = self.sig1(x)
        x = self.fc2(x)
        x = self.sig2(x)
        x = self.fc3(x)
        # x = self.sig3(x)
        # x = self.fc4(x)
        return x


class GRU(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        # batch_size = x.shape[0]
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        # h0 = h0.to(self.device)
        # out, hn = self.gru(x, h0)
        out, hn = self.gru(x)
        # x = self.fc(out[:, -1, :])
        
        # x = self.fc1(hn[-1,:,:])
        # x = self.sig(x)
        x = self.fc2(hn[-1,:,:])
        return x