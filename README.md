# paper

This project implements a multimodal depression prediction model using:
- Text (ALBERT)
- Audio (Wav2Vec2)
- Semantic-Guided Gating (SGCMG)
- Bias-Guided Tensor Product Attention (BG-TPA)
- Emotion Trend Module (ETM)



## 📦 Dataset Format

Expected structure (for both DAIC-WOZ and EATD):

```
data/
├── daic_woz/
│   ├── audio/            
│   ├── transcripts/        
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
```# SBT-Net
