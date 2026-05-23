import torch
import torch.nn as nn


class EmotionTrendModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, audio_feat):
        if audio_feat.ndim == 3:
            audio_feat = audio_feat.mean(dim=1)
        return self.proj(audio_feat)