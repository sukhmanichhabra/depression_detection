import torch
import torch.nn as nn
from transformers import AlbertModel, Wav2Vec2Model
from semantic_gating import SemanticGating
from bias_tensor_attention import BiasGuidedTensorAttention
from emotion_trend_modeling import EmotionTrendModule

class DepressionPredictor(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.text_encoder = AlbertModel.from_pretrained("albert-base-v2")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.sgcmg = SemanticGating(hidden_dim)
        self.bg_tpa = BiasGuidedTensorAttention(hidden_dim)
        self.etm = EmotionTrendModule(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, wav):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        knowledge_feat = text_output[:, 0, :]
        text_gated = self.sgcmg(text_output, knowledge_feat)
        #audio_out = self.audio_encoder(wav.unsqueeze(1)).last_hidden_state
        # 确保 wav shape 为 [B, T]（batch_size, time）
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # 单条数据变成 [1, T]
        elif wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0)  # 防止多余 [1, 1, T]
            wav = wav.unsqueeze(0)  # 重新变回 [1, T]

        # 最终传入 wav2vec2 要是 [batch_size, sequence_length]
        audio_out = self.audio_encoder(wav).last_hidden_state

        audio_bias_attended = self.bg_tpa(audio_out)
        if audio_bias_attended.ndim == 2:
            audio_bias_attended = audio_bias_attended.unsqueeze(1)
        audio_trend = self.etm(audio_out)
        text_gated = text_gated + audio_trend.unsqueeze(1)
        #attn_output, _ = self.cross_attn(text_gated, audio_out, audio_out)
        attn_output, _ = self.cross_attn(text_gated, audio_bias_attended, audio_bias_attended)

        pooled = attn_output.mean(dim=1)
        logits = self.classifier(pooled).squeeze(1)
        return logits