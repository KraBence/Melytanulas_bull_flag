# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
from utils import setup_logger

logger = setup_logger()

def evaluate():
    logger.info("Evaluating model...")

if __name__ == "__main__":
    evaluate()

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Importáljuk a közös modulból!
from config import (OUTPUT_DIR, LABEL_FILE, BATCH_SIZE, INPUT_SIZE)
from utils import (FlagDataset, BaselineLSTM)

MODEL_PATH = os.path.join(OUTPUT_DIR, 'baseline_lstm_best.pth')
CLASSES_PATH = os.path.join(OUTPUT_DIR, 'classes.npy')


def evaluate_model():
    logger("--- BASELINE MODEL KIÉRTÉKELÉSE ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger(f"Eszköz: {device}")

    # 1. Ellenőrzés
    if not os.path.exists(LABEL_FILE):
        logger("HIBA: Nincs label fájl.")
        return
    if not os.path.exists(MODEL_PATH):
        logger("HIBA: Nincs elmentett modell. Futtasd le előbb a 02-training.py-t!")
        return
    if not os.path.exists(CLASSES_PATH):
        logger("HIBA: Nincs classes.npy fájl.")
        return

    # 2. Test Set Regenerálása (Ugyanazzal a logikával)
    df_labels = pd.read_csv(LABEL_FILE)
    allowed = ['EURUSD', 'XAU']
    df_labels = df_labels.dropna(subset=['raw_csv_filename'])
    mask = df_labels['raw_csv_filename'].apply(lambda x: any(a in str(x) for a in allowed))
    df_labels = df_labels[mask].reset_index(drop=True)

    le = LabelEncoder()
    # Csak a transform kell, mert a classokat a fájlból töltjük vissza
    df_labels['label_idx'] = le.fit_transform(df_labels['label'])

    # Betöltjük a mentett osztályneveket
    class_names = np.load(CLASSES_PATH, allow_pickle=True)

    # Split - random_state=42 GARANTÁLJA az egyezést
    train_val, test_df = train_test_split(df_labels, test_size=0.15, stratify=df_labels['label'], random_state=42)

    logger(f"Test Set mérete: {len(test_df)} minta")

    test_loader = DataLoader(FlagDataset(test_df, OUTPUT_DIR), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Modell Betöltése
    model = BaselineLSTM(input_size=INPUT_SIZE, num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. Predikció
    all_preds = []
    all_targets = []

    logger("Predikció futtatása...")
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # 5. Eredmény
    logger("\n" + "=" * 60)
    logger("VÉGLEGES TEST SET EREDMÉNYEK")
    logger("=" * 60)
    logger(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    evaluate_model()