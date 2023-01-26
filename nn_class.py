import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        #self.dropout = nn.Dropout(p=0.1)
        self.hidden1 = torch.nn.Linear(n_hidden, 10)
        self.predict = torch.nn.Linear(10, n_output)
        
        
    def forward(self, x):
        x = nn.functional.relu(self.hidden(x))
        #x = self.dropout(x)
        x = nn.functional.relu(self.hidden1(x))
        x = self.predict(x)
        
        return x


    
