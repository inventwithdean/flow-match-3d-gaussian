import torch 
import torch.nn as nn

class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
    def forward(self, x, t):
        # x: (N, 3) t: (N, 1)
        inputs = torch.cat([x, t], dim=1) # (N, 4)
        return self.net(inputs)