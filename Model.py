import torch
from torch import nn
from torch.autograd import Variable




class Model(nn.Module):

    def __init__(self, size_input, size_hidden_1, size_hidden_2):
        super(Model, self).__init__()
        self.inp = nn.Linear(3, size_hidden_1)
        self.int = nn.Linear(size_hidden_1, size_hidden_2)
        self.out = nn.Linear(size_hidden_2, 1)

    def forward(self, tens_actual_edge):
        out = nn.functional.relu(self.inp(tens_actual_edge))
        out = nn.functional.relu(self.int(out))
        return torch.sigmoid(self.out(out))


'''
class Model(nn.Module):

    def __init__(self, size_input, size_hidden_1, size_hidden_2):
        super(Model, self).__init__()
        self.inp = nn.Linear(size_input, size_hidden_1)
        self.int = nn.Linear(size_hidden_1+3, size_hidden_2)
        self.out = nn.Linear(size_hidden_2, 1)

    def forward(self, tens_edges_occupation, tens_actual_edge):
        out = nn.functional.relu(self.inp(tens_edges_occupation))
        out = nn.functional.relu(self.int(torch.cat((out, tens_actual_edge), dim=-1)))
        return torch.sigmoid(self.out(out))
'''