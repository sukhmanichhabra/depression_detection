# SBT-Net — Architecture Overview

This document summarizes the high-level architecture of SBT-Net (tri-cue multimodal fusion for depression recognition).

## Purpose

Combine textual and audio cues to predict depression using:
- Text: ALBERT encoder
- Audio: lightweight encoder + attention
- Fusion: Semantic-Guided Gating (SGCMG), Bias-Guided Tensor Product Attention (BG-TPA), Emotion Trend Module (ETM)

## Components

- `dataset_loader.py` — `DepressionDataset`: tokenizes transcripts (`AlbertTokenizer`) and loads audio (`librosa`).
- `model.py` — `DepressionPredictor`: overall multimodal model and classifier.
- `semantic_gating.py` — `SemanticGating`: gates text sequence by a knowledge vector.
- `bias_tensor_attention.py` — `BiasGuidedTensorAttention`: computes attention over audio features.
- `emotion_trend_modeling.py` — `EmotionTrendModule`: projects audio into an emotion-trend embedding.
- `train.py` / `test.py` — training and evaluation scripts.

## Data layout

Place datasets under `data/<dataset_name>/` with `audio/`, `transcripts/`, and `labels.csv` (see README).

## Logical dataflow

1. `DepressionDataset` → returns `(input_ids, attention_mask, wav, label)`
2. Text → `AlbertModel` → sequence + pooled knowledge vector
3. Sequence + knowledge → `SemanticGating` → gated text features
4. Audio → resample + adaptive pooling → `audio_encoder` → audio features
5. Audio features → `BiasGuidedTensorAttention` → bias-attended audio
6. Audio features → `EmotionTrendModule` → audio trend vector
7. audio trend added to gated text → cross-attention (`MultiheadAttention`) with bias-attended audio
8. pooled cross-attention output → classifier → scalar logit

## Mermaid diagram

```mermaid
graph LR
  subgraph Input
    T[Transcript Text]
    A[Raw Audio]
  end
  T -->|tokenize| Tokenizer[AlbertTokenizer]
  Tokenizer --> ALBERT[ALBERT Encoder]
  ALBERT --> TextSeq[Text Sequence]
  ALBERT --> Know[Knowledge Vector (CLS)]
  TextSeq --> SemanticGating
  Know --> SemanticGating
  A -->|librosa + pool| AudioPool[Resample & Pool]
  AudioPool --> AudioEnc[Audio Encoder (MLP)]
  AudioEnc --> BGTPA[Bias-Guided Tensor Attention]
  AudioEnc --> ETM[Emotion Trend Module]
  BGTPA --> BiasAudio[Bias-Attended Audio]
  ETM --> AudioTrend[Audio Trend]
  SemanticGating --> GatedText[Gated Text]
  AudioTrend --> AddTrend[Add to Gated Text]
  AddTrend --> CrossAttn[Cross-Attention (text<-audio)]
  BiasAudio --> CrossAttn
  CrossAttn --> Pool[Mean Pool]
  Pool --> Classifier[MLP Classifier]
  Classifier --> Output[Depression Score]
```

## Implementation notes

- Model weights: saved to `saved_models/best_model.pt` by `train.py`.
- Default text model: `albert-base-v2`.
- Audio is limited to 15s (`sr * 15`) in the dataset loader.

## Next steps (suggested)

- Add a diagram image or `ARCHITECTURE.md` to repo root if you want a project-level overview.
- Add shapes/labels to the mermaid diagram for publication figures.

---
Generated from code inspection on 2026-05-22.
