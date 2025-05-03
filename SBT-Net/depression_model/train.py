import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, roc_auc_score
from dataset_loader import DepressionDataset
from model import DepressionPredictor
from tqdm import tqdm
import os



# 验证函数
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for ids, mask, wav, label in loader:
            ids, mask, wav, label = ids.to(device), mask.to(device), wav.to(device), label.to(device)
            output = model(ids, mask, wav)
            loss = criterion(output, label)
            total_loss += loss.item()
            preds = torch.sigmoid(output).cpu().numpy() > 0.5
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    if len(set(all_labels)) < 2:
        auc = 0.5
        print("⚠️ Skipping AUC: only one class present in val set")
    else:
        auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), acc, auc


# 单轮训练函数
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for ids, mask, wav, label in tqdm(loader, desc="Training"):
        ids, mask, wav, label = ids.to(device), mask.to(device), wav.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(ids, mask, wav)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.sigmoid(output).detach().cpu().numpy() > 0.5
        all_preds.extend(preds)
        all_labels.extend(label.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    if len(set(all_labels)) < 2:
        auc = 0.5
        print("⚠️ Skipping AUC: only one class present in train set")
    else:
        auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader1), acc, auc


# 主流程
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = DepressionDataset('/root/autodl-tmp/data/daic_woz/labels.csv', '/root/autodl-tmp/data/daic_woz')

    # 拆分训练集和验证集（使用固定随机种子）
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)

    model = DepressionPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = BCEWithLogitsLoss()

    # Early stopping参数
    best_auc = 0
    patience = 3
    counter = 0
    num_epochs = 20

    os.makedirs('depression_model/saved_models', exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), 'depression_model/saved_models/best_model.pt')
            print("✅ Model saved (Val AUC improved)")
        else:
            counter += 1
            print(f"⏳ No AUC improvement. Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("🛑 Early stopping triggered.")
                break

if __name__ == '__main__':
    main()
