# Multimodal Depression Prediction Model

This project implements a multimodal depression prediction model using:
- Text (ALBERT)
- Audio (Wav2Vec2)
- Semantic-Guided Gating (SGCMG)
- Bias-Guided Tensor Product Attention (BG-TPA)
- Emotion Trend Module (ETM)

## 📁 Project Structure

```
depression_model/
├── dataset_loader.py              # Dataset loading logic
├── semantic_gating.py             # SGCMG module
├── bias_tensor_attention.py       # BG-TPA module
├── emotion_trend_modeling.py      # ETM module
├── model.py                       # Main model architecture
├── train.py                       # Training script
├── test.py                        # Inference + evaluation script
├── preprocess.py                  # Audio preprocessing / alignment
├── requirements.txt               # Dependency list
├── README.md                      # This file
├── .gitignore
```

## 📦 Dataset Format

Expected structure (for both DAIC-WOZ and EATD):

```
data/
├── daic_woz/
│   ├── audio/              # audio/P3001.wav
│   ├── transcripts/        # transcripts/P3001.txt
│   └── labels.csv
├── eatd/
│   ├── audio/
│   ├── transcripts/
│   └── labels.csv
```

### labels.csv format:

```
session_id,text_path,audio_path,label
P3001,transcripts/P3001.txt,audio/P3001.wav,1
...
```

## 🚀 Train

```bash
python train.py
```

## 🧪 Test

```bash
python test.py
```