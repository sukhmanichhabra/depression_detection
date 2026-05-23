import torch
import torch.nn as nn


class BiasGuidedTensorAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, audio_feat):
        if audio_feat.ndim == 2:
            audio_feat = audio_feat.unsqueeze(1)

        query = self.query(audio_feat)
        key = self.key(audio_feat)
        value = self.value(audio_feat)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        return attended.squeeze(1) if attended.size(1) == 1 else attended