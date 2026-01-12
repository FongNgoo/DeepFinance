# news_encoder.py
import torch
import torch.nn as nn

class NewsEncoder(nn.Module):
    def __init__(self, hidden_dim, embed_dim=1024):
        super().__init__()
        self.projector = nn.Linear(embed_dim, hidden_dim)

    def forward(self, s_n):
        # s_n: (B, T, 1024)
        v_n = self.projector(s_n)     # (B, T, hidden_dim)
        return v_n
