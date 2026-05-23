from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from dataset_loader import DepressionDataset
from model import DepressionPredictor
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data' / 'daic_woz'
MODEL_PATH = BASE_DIR / 'saved_models' / 'best_model.pt'


def pad_collate(batch):
    ids, masks, wavs, labels = zip(*batch)
    wavs = pad_sequence(wavs, batch_first=True)
    return torch.stack(ids), torch.stack(masks), wavs, torch.stack(labels)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for ids, mask, wav, label in tqdm(loader):
        ids, mask, wav, label = ids.to(device), mask.to(device), wav.to(device), label.to(device)
        output = model(ids, mask, wav)
        preds = torch.sigmoid(output).cpu().numpy() > 0.5
        all_preds.extend(preds)
        all_labels.extend(label.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    print(f"[Test] Accuracy: {acc:.4f}, AUC: {auc:.4f}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = DepressionDataset(DATA_DIR / 'labels_full.csv', DATA_DIR)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, collate_fn=pad_collate)
    model = DepressionPredictor().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()