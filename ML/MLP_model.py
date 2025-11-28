import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout_p=0.2, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
