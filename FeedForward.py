import torch
import torch.nn as nn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

        self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = self.ln(output + residual)