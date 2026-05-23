"""
model.py  —  SBT-Net with Neuromimetic Acoustic Processing
===========================================================
Changes over baseline
---------------------
* NeuromimeticAcousticProcessor (NAP) extracts Differential Entropy +
  Hjorth Parameters across 5 EEG-inspired sub-bands from the raw waveform.
* The resulting neuro_feat [B, H] is fused with the standard audio pathway
  via a learned gating mechanism before cross-attention.
* Everything else (SemanticGating, BiasGuidedTensorAttention,
  EmotionTrendModule, cross-attention, classifier) is preserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel

from semantic_gating import SemanticGating
from bias_tensor_attention import BiasGuidedTensorAttention
from emotion_trend_modeling import EmotionTrendModule
from neuromimetic_acoustic import NeuromimeticAcousticProcessor


class DepressionPredictor(nn.Module):
    """
    Multimodal depression predictor fusing:
      - ALBERT text encoder  (with SemanticGating)
      - Standard audio MLP encoder  (with BG-TPA + ETM)
      - Neuromimetic Acoustic features  (DE + Hjorth via NAP)
    """

    def __init__(self, hidden_dim: int = 768, sr: int = 16_000):
        super().__init__()
        self.audio_frames = 128

        # ── Text branch ──────────────────────────────────────────────
        self.text_encoder = AlbertModel.from_pretrained("albert-base-v2")
        self.sgcmg        = SemanticGating(hidden_dim)

        # ── Standard audio branch ────────────────────────────────────
        self.audio_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bg_tpa = BiasGuidedTensorAttention(hidden_dim)
        self.etm    = EmotionTrendModule(hidden_dim, hidden_dim)

        # ── Neuromimetic Acoustic branch (NEW) ───────────────────────
        self.nap = NeuromimeticAcousticProcessor(
            sr=sr,
            epoch_sec=0.25,     # 250 ms epochs  (≈ neural oscillation window)
            hop_sec=0.125,      # 50% overlap
            hidden_dim=hidden_dim,
            n_heads=4,
            n_layers=2,
        )

        # Gating to fuse standard audio trend with neuromimetic features
        # gate = σ(W_n · neuro + W_e · etm_trend)  applied to combined vector
        self.neuro_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # ── Cross-attention & classifier ─────────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    # ----------------------------------------------------------------
    def forward(self, input_ids, attention_mask, wav):
        # ── Text encoding ────────────────────────────────────────────
        text_output   = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                             # [B, L, H]
        knowledge_feat = text_output[:, 0, :]          # CLS token [B, H]
        text_gated     = self.sgcmg(text_output, knowledge_feat)  # [B, L, H]

        # ── Standard audio encoding ──────────────────────────────────
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0).unsqueeze(0)

        # Adaptive pool to fixed number of frames
        audio_pooled = F.adaptive_avg_pool1d(
            wav.unsqueeze(1), self.audio_frames
        ).transpose(1, 2)                              # [B, 128, 1]
        audio_out    = self.audio_encoder(audio_pooled.float())  # [B, 128, H]

        # BG-TPA
        audio_bias_attended = self.bg_tpa(audio_out)  # [B, 128, H] or [B, H]
        if audio_bias_attended.ndim == 2:
            audio_bias_attended = audio_bias_attended.unsqueeze(1)  # [B, 1, H]

        # Emotion trend
        audio_trend = self.etm(audio_out)              # [B, H]

        # ── Neuromimetic features (NEW) ──────────────────────────────
        neuro_feat = self.nap(wav)                     # [B, H]

        # Fuse audio_trend and neuro_feat via learned gate
        combined  = torch.cat([audio_trend, neuro_feat], dim=-1)  # [B, 2H]
        gate      = self.neuro_gate(combined)          # [B, H]
        # Weighted combination: gate selects how much of each source to keep
        fused_trend = gate * neuro_feat + (1 - gate) * audio_trend  # [B, H]

        # Add fused trend to gated text (broadcast over sequence length)
        text_gated = text_gated + fused_trend.unsqueeze(1)  # [B, L, H]

        # ── Cross-attention: text queries, audio key/value ───────────
        attn_output, _ = self.cross_attn(
            text_gated, audio_bias_attended, audio_bias_attended
        )                                              # [B, L, H]

        # ── Classify ─────────────────────────────────────────────────
        pooled = attn_output.mean(dim=1)               # [B, H]
        logits = self.classifier(pooled).squeeze(1)    # [B]
        return logits