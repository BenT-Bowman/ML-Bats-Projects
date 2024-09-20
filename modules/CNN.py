import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(3200, 512),
            nn.Linear(512, output_size)
        )
    def forward(self, x):
        return F.softmax(self.model(x))

# class MoE(nn.Module):
#     def __init__(self, output_size, list_len=3):
#         super().__init__()
#         self.experts=nn.ModuleList([CNN(output_size) for _ in range(list_len)])
#         self.gating_network = CNN(list_len)
#     def forward(self, x):
#         gating_weights=self.gating_network(x)
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)

#         output = torch.einsum('boe,be->bo', expert_outputs, gating_weights)
#         return output
    
