import torch
import torch.nn as nn


def custom_model(nn.Module):
    def __init__(self):
        super.__init__(self)
        self.fc1 = nn.Linear(10,8)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(8,5)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(5,2)
        self.sig = nn.Sigmoid()
    
    def forward(self,x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.sig(self.fc3(x))

    def create_network(json_config=None):
        return 