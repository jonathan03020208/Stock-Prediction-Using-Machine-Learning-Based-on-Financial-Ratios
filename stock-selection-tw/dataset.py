from torch.utils.data import Dataset
import torch


class dataset(Dataset):
    def __init__(self, model_type, ratio, quarterly_return, index_list):
        self.X = []
        self.Y = []
        if model_type == 'FNN':
            for i in range(len(ratio)):
                for k in index_list:
                    self.X.append(ratio[i][k - 1])         
                    self.Y.append(quarterly_return[i][k - 1])
        elif model_type == 'GRU':
            for i in range(len(ratio)):
                for k in index_list:
                    self.X.append(ratio[i][k - 1:k + 3][:])
                    self.Y.append(quarterly_return[i][k - 1])
        else:
            raise RuntimeError("model_type error")
        
    def __getitem__(self, idx):
        return (torch.Tensor(self.X[idx]), torch.Tensor([self.Y[idx]]))
    
    def __len__(self):
        return len(self.X)