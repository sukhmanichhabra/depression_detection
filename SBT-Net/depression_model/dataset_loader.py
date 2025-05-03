import os
import torch
import librosa
import pandas as pd
from torch.utils.data import Dataset
from transformers import AlbertTokenizer

class DepressionDataset(Dataset):
    def __init__(self, label_file, base_dir, tokenizer_name="albert-base-v2", max_len=128, sr=16000):
        self.df = pd.read_csv(label_file)
        self.base_dir = base_dir
        self.tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.sr = sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_path = os.path.join(self.base_dir, row['text_path'])
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        audio_path = os.path.join(self.base_dir, row['audio_path'])
        wav, _ = librosa.load(audio_path, sr=self.sr)
        #wav = torch.tensor(wav[:self.sr * 15])  # 限制最多15秒
        wav = torch.tensor(wav[:self.sr * 15], dtype=torch.float32)
        label = torch.tensor(float(row['label']), dtype=torch.float)
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), wav, label