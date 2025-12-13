import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
import config
from utils import setup_logger, FlagDataset, BaselineLSTM, HybridModel, EnsembleModel

# Kezdeti logger (később felülírjuk fájlba írással a main-ben)
logger = setup_logger()


def get_test_loader(batch_size, seq_len):
    """
    Létrehozza a Test DataLoadert (ugyanazzal a seeddel, mint a tréningnél!)
    """
    label_path = config.LABEL_FILE
    if not os.path.exists(label_path):
        # Fallback keresés
        alt = os.path.join(config.DATA_ROOT, "ground_truth_labels.csv")
        if os.path.exists(alt):
            label_path = alt
        else:
            return None, None, None

    df_labels = pd.read_csv(label_path)
    df_labels = df_labels.dropna(subset=['clean_csv_filename'])

    le = LabelEncoder()
    df_labels['label_idx'] = le.fit_transform(df_labels['label'])

    # Betöltjük a tréning során mentett osztályokat
    classes_path = os.path.join(config.OUTPUT_DIR, 'classes.npy')
    if os.path.exists(classes_path):
        class_names = np.load(classes_path, allow_pickle=True)
    else:
        class_names = le.classes_

    # Split (fix seed 42!)
    train_val, test_df = train_test_split(df_labels, test_size=0.15, stratify=df_labels['label'], random_state=42)

    ds = FlagDataset(test_df, config.DATA_ROOT, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    return loader, class_names, len(test_df)


def evaluate_single_model(model_name, ModelClass, input_size, seq_len, batch_size):
    """
    Egyetlen modell kiértékelése.
    """
    logger.info(f"\n--- {model_name.upper()} KIÉRTÉKELÉSE ---")

    # 1. Adatbetöltés (Model specifikus paraméterekkel)
    test_loader, class_names, n_samples = get_test_loader(batch_size, seq_len)
    if test_loader is None:
        logger.error("Hiba az adatok betöltésekor.")
        return

    logger.info(f"Test Set: {n_samples} minta")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Modell betöltése
    model_path = os.path.join(config.OUTPUT_DIR, f'{model_name}_best.pth')
    if not os.path.exists(model_path):
        logger.warning(f"SKIPPING: Nem található a modell fájl: {model_path}")
        return

    model = ModelClass(input_size=input_size, num_classes=len(class_names))

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Hiba a modell betöltésekor: {e}")
        return

    # 3. Predikció
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # 4. Report & Confusion Matrix
    logger.info(f"\nEREDMÉNYEK: {model_name}")
    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    logger.info("\n" + report)

    # --- CONFUSION MATRIX KIÍRÁSA ---
    cm = confusion_matrix(all_targets, all_preds)
    # DataFrame-be rakjuk, hogy látszódjanak az osztálynevek a logban
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    logger.info("\nCONFUSION MATRIX:\n" + str(cm_df))


if __name__ == "__main__":
    # --- LOGOLÁS BEÁLLÍTÁSA ---
    # Nem generálunk egyedi fájlnevet, a utils.py megoldja a közös run.log-ot
    logger = setup_logger()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Fejléc logolása (hogy látszódjon mikor indult ez a futás)
    logger.info("\n" + "=" * 60)
    logger.info(f"KIÉRTÉKELÉS INDÍTÁSA: {current_time}")
    logger.info(f"Log fájl helye: {config.LOG_FILE}")
    logger.info("=" * 60)

    # 1. BASELINE LSTM KIÉRTÉKELÉS
    evaluate_single_model(
        model_name="baseline_lstm",
        ModelClass=BaselineLSTM,
        input_size=4,
        seq_len=50,
        batch_size=32
    )

    # 2. HYBRID MODEL KIÉRTÉKELÉS
    evaluate_single_model(
        model_name="hybrid_model",
        ModelClass=HybridModel,
        input_size=config.INPUT_SIZE,
        seq_len=config.SEQUENCE_LENGTH,
        batch_size=config.BATCH_SIZE
    )

    # 3. ENSEMBLE MODEL KIÉRTÉKELÉS
    # evaluate_single_model(
    #     model_name="ensemble_model",
    #     ModelClass=EnsembleModel,
    #     input_size=config.INPUT_SIZE,
    #     seq_len=config.SEQUENCE_LENGTH,
    #     batch_size=config.BATCH_SIZE
    # )