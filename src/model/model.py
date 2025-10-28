import torch
import torch.nn as nn
import torch.optim as optim

class CancerRiskModel(nn.Module):
    def __init__(self, input_dim):
        super(CancerRiskModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def create_model(input_dim):
    model = CancerRiskModel(input_dim)
    return model