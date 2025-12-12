# Model training script
# This script defines the model architecture and runs the training loop.
import config
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils import setup_logger
logger = setup_logger()
# Importáljuk a közös modulból!
from config import (OUTPUT_DIR, LABEL_FILE, BATCH_SIZE, INPUT_SIZE)
from utils import (FlagDataset, BaselineLSTM)

# Training specifikus paraméterek
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001
CLIP_VALUE = 1.0


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

    # Mentjük az osztályokat
    np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), le.classes_)

    # Split
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

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_path = os.path.join(OUTPUT_DIR, f'{model_name}_best.pth')
            torch.save(model.state_dict(), save_path)
            logger(f"  -> Modell mentve (Acc: {v_acc:.1f}%)")

        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger(f"[STOP] Early Stopping {EARLY_STOPPING_PATIENCE} epoch után.")
                break

    logger("\n[INFO] Tanítás kész.")


if __name__ == "__main__":
    data = prepare_data(LABEL_FILE, OUTPUT_DIR)
    if data:
        model = BaselineLSTM(INPUT_SIZE=INPUT_SIZE, num_classes=data['num_classes'])
        train_engine(model, data)