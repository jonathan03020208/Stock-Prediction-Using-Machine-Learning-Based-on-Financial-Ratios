import torch.nn as nn
import torch
from torch_geometric.nn import GATConv, GATv2Conv
import pandas as pd


input = pd.read_csv('ratio-2022-12-13.csv', dtype='str', keep_default_na=False)

sector_dict = {
    'Technology': 0,
    'Energy': 1,
    'Consumer Cyclical': 2,
    'Industrials': 3,
    'Consumer Defensive': 4,
    'Real Estate': 5,
    'Financial Services': 6,
    'Communication Services': 7,
    'Basic Materials': 8
}

NUMBER_OF_COMPANY = 97
number_of_quarters_per_stock = 37

sector_list = []
for i in range(0, number_of_quarters_per_stock * NUMBER_OF_COMPANY, number_of_quarters_per_stock):
    sector_list.append(sector_dict[input['Sector'][i]])


class GAT(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, intra_edge_index, inter_edge_index, cat, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout)
        self.intra_gat = GATConv(hidden_size, hidden_size, dropout=dropout)
        self.inter_gat = GATConv(hidden_size, hidden_size, dropout=dropout)
        self.intra_edge_index = intra_edge_index
        self.inter_edge_index = inter_edge_index

        self.cat = cat
        if cat == 1:
            self.fusion = nn.Linear(3 * hidden_size, 1)
        elif cat == 2:
            self.fusion = nn.Linear(hidden_size, 1)
        else:
            self.fusion = nn.Linear(2 * hidden_size, 1)

        nn.init.xavier_uniform_(self.fusion.weight, gain=nn.init.calculate_gain('linear'))
        # self.reg_layer = nn.Linear(hidden_size,1)


    def forward(self, x):
        # batch_size = x.shape[0]
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        # h0 = h0.to(self.device)
        # out, hn = self.gru(x, h0)
        out, hn = self.gru(x)
        # x = self.fc(out[:, -1, :])
        # x = self.fc1(hn[-1,:,:])
        seq_emb = hn[-1,:,:]
        intra_emb = self.intra_gat(seq_emb, self.intra_edge_index)

        sector_vectors = [[] for _ in range(len({s for s in sector_list}))]
        for stock_num, embedding_vec in zip(range(NUMBER_OF_COMPANY), intra_emb):
            sector_num = sector_list[stock_num]
            sector_vectors[sector_num].append(embedding_vec.tolist())
        max_pool_result = []
        for z in sector_vectors:
            # max_pool_result.append([0.0 for y in zip(*z)])
            max_pool_result.append([max(y)for y in zip(*z)])
            # max_pool_result.append([sum(y) / len(y) for y in zip(*z)])

        if self.cat == 1 or self.cat == 3:
            inter_emb = self.inter_gat(torch.Tensor(max_pool_result), self.inter_edge_index)
            inter_emb_temp = []
            for stock_num in range(NUMBER_OF_COMPANY):
                sector_num = sector_list[stock_num]
                inter_emb_temp.append(inter_emb[sector_num].tolist())
            inter_emb = torch.Tensor(inter_emb_temp)

        #full (1)
        if self.cat == 1:
            fusion_emb = torch.cat((seq_emb, intra_emb, inter_emb), dim=1)

        # 2 : only gru(w/o intra and inter)\
        elif self.cat == 2:
            fusion_emb = seq_emb

        # 3 : w/o intra
        elif self.cat == 3:
            fusion_emb = torch.cat((seq_emb, inter_emb), dim=1)

        # 4 : w/o inter
        else:
            fusion_emb = torch.cat((seq_emb, intra_emb), dim=1)
        
        output = self.fusion(fusion_emb)
        # fusion_emb = torch.sigmoid(self.fusion(fusion_emb))

        # output = self.reg_layer(fusion_emb)
        output = torch.flatten(output)
        return output