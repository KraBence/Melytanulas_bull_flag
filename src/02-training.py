# Model training script
# This script defines the model architecture and runs the training loop.
import config
from utils import setup_logger

logger = setup_logger()


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- KONFIGURÁCIÓ ---
OUTPUT_DIR = "/app/output"
if not os.path.exists(OUTPUT_DIR): OUTPUT_DIR = "./output"

LABEL_FILE = os.path.join(OUTPUT_DIR, 'ground_truth_labels.csv')

# --- HIPERPARAMÉTEREK ---
BATCH_SIZE = 32
SEQUENCE_LENGTH = 50
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 30
CLIP_VALUE = 1.0    # ÚJ: Gradiens vágás értéke (Critical for LSTM)

# --- 1. DATASET OSZTÁLY (Biztonságosabb) ---
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

            # --- JAVÍTÁS 1: Biztonságos Normalizálás ---
            seg_min = segment.min(axis=0)
            seg_max = segment.max(axis=0)

            # Hozzáadunk egy pici számot (1e-6), hogy sose osszunk nullával
            denom = seg_max - seg_min + 1e-6
            segment_norm = (segment - seg_min) / denom

            # --- JAVÍTÁS 2: NaN ellenőrzés ---
            # Ha bármi NaN lett (pl. végtelen adat miatt), cseréljük nullára
            segment_norm = np.nan_to_num(segment_norm, nan=0.0, posinf=1.0, neginf=0.0)

            tensor_segment = torch.tensor(segment_norm, dtype=torch.float32).permute(1, 0).unsqueeze(0)
            tensor_resized = torch.nn.functional.interpolate(
                tensor_segment, size=self.seq_len, mode='linear', align_corners=False
            )
            tensor_final = tensor_resized.squeeze(0).permute(1, 0)

            # Végső ellenőrzés: ha még mindig van NaN a tensorban
            if torch.isnan(tensor_final).any():
                return self._get_dummy()

            return tensor_final, torch.tensor(row['label_idx'], dtype=torch.long)

        except Exception:
            return self._get_dummy()

    def _get_dummy(self):
        # Biztonsági kimenet hiba esetén
        return torch.zeros((self.seq_len, INPUT_SIZE)), torch.tensor(0, dtype=torch.long)

# --- 2. MODELL ---
class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits

# --- 3. MAIN ---
def main():
    print("--- BASELINE MODEL TRAINING (LSTM 2-Layer) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(LABEL_FILE):
        print(f"HIBA: {LABEL_FILE} nem található.")
        return
    df_labels = pd.read_csv(LABEL_FILE)

    # Szűrés
    allowed = ['EURUSD', 'XAU']
    df_labels = df_labels.dropna(subset=['raw_csv_filename'])
    mask = df_labels['raw_csv_filename'].apply(lambda x: any(a in str(x) for a in allowed))
    df_labels = df_labels[mask].reset_index(drop=True)
    print(f"Adatok száma: {len(df_labels)}")

    if len(df_labels) < 10:
        print("Túl kevés adat!")
        return

    le = LabelEncoder()
    df_labels['label_idx'] = le.fit_transform(df_labels['label'])
    num_classes = len(le.classes_)
    print(f"Osztályok: {le.classes_}")
    np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), le.classes_)

    # Split
    train_val_df, test_df = train_test_split(
        df_labels, test_size=0.15, random_state=42, stratify=df_labels['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.176, random_state=42, stratify=train_val_df['label']
    )

    # DataLoaders
    train_ds = FlagDataset(train_df, OUTPUT_DIR, SEQUENCE_LENGTH)
    val_ds = FlagDataset(val_df, OUTPUT_DIR, SEQUENCE_LENGTH)
    test_ds = FlagDataset(test_df, OUTPUT_DIR, SEQUENCE_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Init
    model = BaselineLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nIndul a tanítás...")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

            # --- JAVÍTÁS 3: GRADIENT CLIPPING ---
            # Ez akadályozza meg, hogy a Loss NaN legyen!
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_acc = 100 * correct / total if total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'baseline_model.pth'))

    # Final Test
    print("\n--- VÉGSŐ TESZT ---")
    if os.path.exists(os.path.join(OUTPUT_DIR, 'baseline_model.pth')):
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'baseline_model.pth'), weights_only=True))

    model.eval()
    test_correct = 0
    test_total = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    print(f"Teszt Pontosság: {100 * test_correct / test_total:.2f}%")
    print(classification_report(all_targets, all_preds, target_names=le.classes_, zero_division=0))

if __name__ == "__main__":
    main()

def train():
    logger.info("Starting training process...")
    logger.info(f"Loaded configuration. Epochs: {config.EPOCHS}")

    # Simulation of training loop
    for epoch in range(1, config.EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{config.EPOCHS} - Training...")

    logger.info("Training complete.")




