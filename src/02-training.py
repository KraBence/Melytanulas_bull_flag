# Model training script
# This script defines the model architecture and runs the training loop.
import config
from utils import setup_logger
logger = setup_logger()

import os
# --- KONFIGURÁCIÓ ---
OUTPUT_DIR = "/app/output"
if not os.path.exists(OUTPUT_DIR): OUTPUT_DIR = "./output"
LABEL_FILE = os.path.join(OUTPUT_DIR, 'ground_truth_labels.csv')

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 0. GLOBÁLIS KONFIGURÁCIÓ
# ==========================================
OUTPUT_DIR = "/app/output"
if not os.path.exists(OUTPUT_DIR): OUTPUT_DIR = "./output"
LABEL_FILE = os.path.join(OUTPUT_DIR, 'ground_truth_labels.csv')

# Training Paraméterek
BATCH_SIZE = 32
SEQUENCE_LENGTH = 50
INPUT_SIZE = 4
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001
CLIP_VALUE = 1.0

# Modell Paraméterek
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


# ==========================================
# 1. OSZTÁLYOK (Dataset & Model)
# ==========================================

class FlagDataset(Dataset):
    def __init__(self, metadata, csv_dir, seq_len=50):
        self.metadata = metadata
        self.csv_dir = csv_dir
        self.seq_len = seq_len
        self.loaded_dfs = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uniform_name = row['raw_csv_filename']
        csv_filename = f"merged_{uniform_name}.csv"
        csv_path = os.path.join(self.csv_dir, csv_filename)

        if uniform_name not in self.loaded_dfs:
            if os.path.exists(csv_path):
                try:
                    self.loaded_dfs[uniform_name] = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                except:
                    self.loaded_dfs[uniform_name] = None
            else:
                self.loaded_dfs[uniform_name] = None

        df = self.loaded_dfs[uniform_name]
        if df is None: return self._get_dummy()

        try:
            start_ts = row['pole_start_ts'] if pd.notna(row.get('pole_start_ts')) else row['flag_start_ts']
            end_ts = row['flag_end_ts']
            mask = (df.index >= pd.to_datetime(start_ts)) & (df.index <= pd.to_datetime(end_ts))
            segment = df.loc[mask, ['open', 'high', 'low', 'close']].values

            if len(segment) < 2: return self._get_dummy()

            seg_min = segment.min(axis=0)
            seg_max = segment.max(axis=0)
            denom = seg_max - seg_min + 1e-6
            segment_norm = (segment - seg_min) / denom
            segment_norm = np.nan_to_num(segment_norm, nan=0.0, posinf=1.0, neginf=0.0)

            tensor_segment = torch.tensor(segment_norm, dtype=torch.float32).permute(1, 0).unsqueeze(0)
            tensor_resized = torch.nn.functional.interpolate(tensor_segment, size=self.seq_len, mode='linear',
                                                             align_corners=False)
            tensor_final = tensor_resized.squeeze(0).permute(1, 0)

            if torch.isnan(tensor_final).any(): return self._get_dummy()
            return tensor_final, torch.tensor(row['label_idx'], dtype=torch.long)

        except:
            return self._get_dummy()

    def _get_dummy(self):
        return torch.zeros((self.seq_len, INPUT_SIZE)), torch.tensor(0, dtype=torch.long)


class BaselineLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(HIDDEN_SIZE * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits


# ==========================================
# 2. ADATELŐKÉSZÍTÉS & TRAINER
# ==========================================

def prepare_data(label_path, output_dir):
    logger("\n[1] ADATELŐKÉSZÍTÉS (TRAIN/VAL)...")
    if not os.path.exists(label_path): return None
    df_labels = pd.read_csv(label_path)

    allowed = ['EURUSD', 'XAU']
    df_labels = df_labels.dropna(subset=['raw_csv_filename'])
    mask = df_labels['raw_csv_filename'].apply(lambda x: any(a in str(x) for a in allowed))
    df_labels = df_labels[mask].reset_index(drop=True)
    if len(df_labels) < 32: return None

    le = LabelEncoder()
    df_labels['label_idx'] = le.fit_transform(df_labels['label'])

    # Mentjük az osztályokat a kiértékeléshez!
    np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), le.classes_)

    # Split (42-es seed FONTOS, hogy a teszt script ugyanazt vágja le!)
    train_val, test = train_test_split(df_labels, test_size=0.15, stratify=df_labels['label'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, stratify=train_val['label'], random_state=42)

    y_train = train['label_idx'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    train_loader = DataLoader(FlagDataset(train, output_dir), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FlagDataset(val, output_dir), batch_size=BATCH_SIZE, shuffle=False)

    return {'train': train_loader, 'val': val_loader, 'weights': class_weights, 'num_classes': len(le.classes_)}


def train_engine(model, data_package, model_name="baseline_lstm"):
    logger(f"\n[2] {model_name.upper()} TANÍTÁSA INDUL...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weights_tensor = torch.tensor(data_package['weights'], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        for X, y in data_package['train']:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # Validation
        model.eval()
        val_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for X, y in data_package['val']:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                v_total += y.size(0)
                v_correct += (predicted == y).sum().item()

        avg_t_loss = train_loss / len(data_package['train'])
        avg_v_loss = val_loss / len(data_package['val'])
        t_acc = 100 * correct / total
        v_acc = 100 * v_correct / v_total

        scheduler.step(avg_v_loss)

        logger(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_t_loss:.4f}/{avg_v_loss:.4f} | Acc: {t_acc:.1f}%/{v_acc:.1f}%")

        # Checkpoint
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_path = os.path.join(OUTPUT_DIR, f'{model_name}_best.pth')
            torch.save(model.state_dict(), save_path)
            logger(f"  -> Modell mentve (Acc: {v_acc:.1f}%)")

        # Early Stopping
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger(f"[STOP] Early Stopping {EARLY_STOPPING_PATIENCE} epoch után.")
                break

    logger("\n[INFO] Tanítás kész. A tesztelést a másik scripttel futtasd!")


if __name__ == "__main__":
    data = prepare_data(LABEL_FILE, OUTPUT_DIR)
    if data:
        model = BaselineLSTM(INPUT_SIZE=INPUT_SIZE, num_classes=data['num_classes'])
        train_engine(model, data)