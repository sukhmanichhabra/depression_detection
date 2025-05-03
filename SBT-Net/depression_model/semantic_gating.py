import torch
import torch.nn as nn

class SemanticGating(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_t = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_feat, knowledge_feat):
        t = self.linear_t(text_feat.mean(dim=1))
        k = self.linear_k(knowledge_feat)
        gate = self.sigmoid(t * k)
        gated_text = text_feat * gate.unsqueeze(1)
        return gated_text