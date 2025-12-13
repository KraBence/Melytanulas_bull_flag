import sys
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

import config
from utils import setup_logger, FlagDataset, BaselineLSTM, HybridModel, EnsembleModel, FocalLoss

logger = setup_logger()


# FIX BASELINE PARAMÉTEREK
BASELINE_BATCH_SIZE = 32
BASELINE_SEQ_LEN = 50
BASELINE_INPUT_SIZE = 4


# FÜGGVÉNYEK

def prepare_data(label_path, data_root, output_dir, batch_size, seq_len):
    """Adatok betöltése és előkészítése."""
    logger.info(f"\n[1] DATA PREPARATION (BS={batch_size}, SEQ={seq_len})...")

    if not os.path.exists(label_path):
        logger.error(f"ERROR: Label file not found: {label_path}")
        return None

    df_labels = pd.read_csv(label_path)
    df_labels = df_labels.dropna(subset=['clean_csv_filename'])

    if len(df_labels) < 32:
        logger.error("ERROR: Not enough data!")
        return None

    # Encoding
    le = LabelEncoder()
    df_labels['label_idx'] = le.fit_transform(df_labels['label'])

    # Osztálynevek mentése (csak egyszer kell)
    classes_path = os.path.join(output_dir, 'classes.npy')
    np.save(classes_path, le.classes_)

    # Split
    train_val, test = train_test_split(df_labels, test_size=0.15, stratify=df_labels['label'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, stratify=train_val['label'], random_state=42)

    # Weights
    y_train = train['label_idx'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    # Loaders
    train_ds = FlagDataset(train, csv_dir=data_root, seq_len=seq_len)
    val_ds = FlagDataset(val, csv_dir=data_root, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader, 'val': val_loader,
        'weights': class_weights, 'num_classes': len(le.classes_)
    }


def train_engine(model, data_package, model_name):
    """Általános tréning loop."""
    logger.info(f"\n[2] TRAINING START: {model_name.upper()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weights_tensor = torch.tensor(data_package['weights'], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        for X, y in data_package['train']:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_VALUE)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # Val
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

        # Stats
        avg_t_loss = train_loss / len(data_package['train'])
        avg_v_loss = val_loss / len(data_package['val'])
        t_acc = 100 * correct / total if total > 0 else 0
        v_acc = 100 * v_correct / v_total if v_total > 0 else 0

        scheduler.step(avg_v_loss)

        logger.info(
            f"{model_name} | Ep {epoch + 1}/{config.EPOCHS} | L: {avg_t_loss:.4f}/{avg_v_loss:.4f} | Acc: {t_acc:.1f}%/{v_acc:.1f}%")

        # Save Best
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_path = os.path.join(config.OUTPUT_DIR, f'{model_name}_best.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f"  -> Saved Best {model_name} (Acc: {v_acc:.1f}%)")

        # Early Stopping
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"[STOP] Early Stopping: {model_name}")
                break



if __name__ == "__main__":
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    # 1. MODELL: BASELINE LSTM (Fix paraméterekkel)

    # logger.info("\n=== 1. BASELINE LSTM INDÍTÁSA ===")
    # data_baseline = prepare_data(
    #     config.LABEL_FILE, config.DATA_ROOT, config.OUTPUT_DIR,
    #     batch_size=BASELINE_BATCH_SIZE, seq_len=BASELINE_SEQ_LEN
    # )
    # if data_baseline:
    #     model_bl = BaselineLSTM(input_size=BASELINE_INPUT_SIZE, num_classes=data_baseline['num_classes'])
    #     train_engine(model_bl, data_baseline, model_name="baseline_lstm")


    # 2. MODELL: HYBRID (CNN-Transformer) (Config paraméterekkel)

    # logger.info("\n=== 2. HYBRID MODELL INDÍTÁSA ===")
    # data_hybrid = prepare_data(
    #     config.LABEL_FILE, config.DATA_ROOT, config.OUTPUT_DIR,
    #     batch_size=config.BATCH_SIZE, seq_len=config.SEQUENCE_LENGTH
    # )
    # if data_hybrid:
    #     model_hybrid = HybridModel(input_size=config.INPUT_SIZE, num_classes=data_hybrid['num_classes'])
    #     train_engine(model_hybrid, data_hybrid, model_name="hybrid_model")

    # 3. MODELL: ENSEMBLE
    logger.info("\n=== 3. ENSEMBLE MODEL (LSTM + HYBRID) INDÍTÁSA ===")

    # Adatok betöltése
    data_ens = prepare_data(
        config.LABEL_FILE, config.DATA_ROOT, config.OUTPUT_DIR,
        batch_size=config.BATCH_SIZE, seq_len=config.SEQUENCE_LENGTH
    )

    if data_ens:
        # Ensemble inicializálás
        model_ens = EnsembleModel(
            input_size=config.INPUT_SIZE,
            num_classes=data_ens['num_classes']
        )

        # Tanítás indítása
        train_engine(model_ens, data_ens, model_name="ensemble_model")