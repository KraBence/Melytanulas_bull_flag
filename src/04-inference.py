import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import requests
import zipfile

# Útvonal beállítása
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import config
    from utils import setup_logger, BaselineLSTM, HybridModel, EnsembleModel
except ImportError:
    import config
    from utils import setup_logger, BaselineLSTM, HybridModel, EnsembleModel

logger = setup_logger()

# ==========================================
# KONFIGURÁCIÓ INFERENCE-HEZ
# ==========================================
# Link a fájlhoz (ZIP vagy CSV)


MODEL_TYPE = "hybrid"  # "ensemble", "hybrid", "baseline"
CONFIDENCE_THRESHOLD = 0.80


# ==========================================
# 1. LETÖLTŐ FÜGGVÉNY (ÚJ RÉSZ)
# ==========================================
def download_and_prepare_data(url, target_dir):
    """
    Letölti az adatot a megadott URL-ről.
    Megpróbálja ZIP-ként kicsomagolni. Ha nem ZIP, akkor CSV-ként menti.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    temp_file = os.path.join(target_dir, "downloaded_data.tmp")

    logger.info(f"Letöltés indítása innen: {url}")

    try:
        # 1. Letöltés streamelve (hogy ne egye meg a RAM-ot nagy fájlnál)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = 0
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)

        size_mb = total_size / (1024 * 1024)
        logger.info(f"Letöltés kész! Méret: {size_mb:.2f} MB")

        # 2. Megpróbáljuk ZIP-ként kicsomagolni
        try:
            logger.info("Próba: ZIP kicsomagolása...")
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            logger.info(f"Sikeresen kicsomagolva ide: {target_dir}")
            os.remove(temp_file)  # Töröljük a temp zipet

        except zipfile.BadZipFile:
            # Ha nem ZIP, akkor feltételezzük, hogy ez maga a CSV
            logger.info("A fájl nem ZIP. Átnevezés CSV-re...")
            final_csv = os.path.join(target_dir, "downloaded_inference.csv")
            if os.path.exists(final_csv):
                os.remove(final_csv)
            os.rename(temp_file, final_csv)
            logger.info(f"Fájl mentve: {final_csv}")

        return target_dir

    except Exception as e:
        logger.error(f"Hiba a letöltés során: {e}")
        return None


# ==========================================
# 2. INFERENCE DATASET
# ==========================================
class InferenceDataset(Dataset):
    def __init__(self, df, seq_len):
        self.df = df
        self.seq_len = seq_len
        # Csak a szükséges oszlopok
        self.data = df[['open', 'high', 'low', 'close']].values
        self.timestamps = df.index

    def __len__(self):
        return max(0, len(self.df) - self.seq_len)

    def __getitem__(self, idx):
        segment = self.data[idx: idx + self.seq_len]

        # Normalizálás
        seg_min = segment.min(axis=0)
        seg_max = segment.max(axis=0)
        denom = seg_max - seg_min + 1e-6
        segment_norm = (segment - seg_min) / denom
        segment_norm = np.nan_to_num(segment_norm, nan=0.0, posinf=1.0, neginf=0.0)

        tensor = torch.tensor(segment_norm, dtype=torch.float32)
        end_ts = self.timestamps[idx + self.seq_len - 1]

        return tensor, str(end_ts)


# ==========================================
# 3. MODELL BETÖLTÉSE
# ==========================================
def load_trained_model(model_type, input_size, num_classes, device):
    logger.info(f"Modell betöltése: {model_type.upper()}...")

    if model_type == "baseline":
        model = BaselineLSTM(input_size, num_classes)
        fname = "baseline_lstm_best.pth"
    elif model_type == "hybrid":
        model = HybridModel(input_size, num_classes)
        fname = "hybrid_model_best.pth"
    elif model_type == "ensemble":
        model = EnsembleModel(input_size, num_classes)
        fname = "ensemble_model_best.pth"
    else:
        raise ValueError(f"Ismeretlen model_type: {model_type}")

    model_path = os.path.join(config.OUTPUT_DIR, fname)
    if not os.path.exists(model_path):
        logger.error(f"NEM TALÁLHATÓ A MODELL FÁJL: {model_path}")
        logger.error("Futtasd le előbb a 02-training.py-t!")
        sys.exit(1)

    try:
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)

        # Smart load (ha esetleg méreteltérés lenne a PositionalEncodingnál)
        model_state = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Hiba a modell súlyok betöltésekor: {e}")
        sys.exit(1)


# ==========================================
# 4. KERESÉS (SCANNING)
# ==========================================
def scan_file(file_path, model, class_names, device):
    filename = os.path.basename(file_path)
    logger.info(f"Fájl elemzése: {filename}")

    # CSV Betöltése
    try:
        df = pd.read_csv(file_path)

        # Idő oszlop keresése és beállítása
        t_col = next((c for c in df.columns if 'time' in c or 'date' in c), None)
        if t_col:
            # Automatikus dátum felismerés
            try:
                df[t_col] = pd.to_datetime(df[t_col], unit='ms' if df[t_col].apply(
                    lambda x: isinstance(x, (int, float)) and x > 30000000000).any() else None)
            except:
                df[t_col] = pd.to_datetime(df[t_col])

            df = df.set_index(t_col).sort_index()
        else:
            logger.warning(f"  [SKIP] Nincs 'time' vagy 'date' oszlop: {filename}")
            return []

        required = ['open', 'high', 'low', 'close']
        if not all(c in df.columns for c in required):
            logger.warning(f"  [SKIP] Hiányzó OHLC oszlopok: {filename}")
            return []

    except Exception as e:
        logger.error(f"  [SKIP] Hiba a fájl olvasásakor: {e}")
        return []

    if len(df) < config.SEQUENCE_LENGTH:
        logger.warning(f"  [SKIP] Túl rövid fájl (< {config.SEQUENCE_LENGTH} sor)")
        return []

    # Dataset és DataLoader
    ds = InferenceDataset(df, config.SEQUENCE_LENGTH)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    found_patterns = []

    with torch.no_grad():
        for batch_X, batch_ts in loader:
            batch_X = batch_X.to(device)

            logits = model(batch_X)
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            mask = max_probs > CONFIDENCE_THRESHOLD

            if mask.any():
                indices = torch.nonzero(mask).squeeze()
                if indices.ndim == 0: indices = indices.unsqueeze(0)

                for idx in indices:
                    prob = max_probs[idx].item()
                    label_idx = preds[idx].item()
                    timestamp = batch_ts[idx]
                    class_name = class_names[label_idx]

                    found_patterns.append({
                        'Source_File': filename,
                        'Timestamp': timestamp,
                        'Pattern': class_name,
                        'Confidence': round(prob * 100, 2)
                    })

    return found_patterns


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    logger.info("=== 04 - INFERENCE (LETÖLTÉS ÉS KERESÉS) ===")

    # 1. Osztálynevek betöltése
    classes_path = os.path.join(config.OUTPUT_DIR, 'classes.npy')
    if not os.path.exists(classes_path):
        logger.error("Nincs classes.npy! Futtasd le a 02-training.py-t!")
        sys.exit(1)
    class_names = np.load(classes_path, allow_pickle=True)
    logger.info(f"Osztályok: {class_names}")

    # 2. Modell betöltése
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(MODEL_TYPE, config.INPUT_SIZE, len(class_names), device)

    # 3. ADATOK LETÖLTÉSE
    download_dir = os.path.join(config.DATA_ROOT, "inference_downloads")
    prepared_dir = download_and_prepare_data(config.INFERENCE_URL, download_dir)

    if not prepared_dir:
        logger.error("A letöltés sikertelen volt. Kilépés.")
        sys.exit(1)

    # 4. Fájlok keresése a letöltött mappában
    files = glob.glob(os.path.join(prepared_dir, "*.csv"))

    if not files:
        # Ha esetleg almappába csomagolta ki a ZIP
        files = glob.glob(os.path.join(prepared_dir, "**", "*.csv"), recursive=True)

    logger.info(f"Feldolgozandó CSV fájlok száma: {len(files)}")

    all_results = []

    # 5. Futtatás
    for i, f in enumerate(files):
        # macOS rejtett fájlok kiszűrése
        if os.path.basename(f).startswith("._"): continue

        print(f"[{i + 1}/{len(files)}] Elemzés: {os.path.basename(f)} ...", end="\r")
        patterns = scan_file(f, model, class_names, device)
        if patterns:
            all_results.extend(patterns)
            logger.info(f"\n  -> Találat: {len(patterns)} db alakzat.")

    # 6. Eredmények mentése
    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df = res_df.sort_values(by=['Source_File', 'Timestamp'])

        save_path = os.path.join(config.OUTPUT_DIR, 'inference_results.csv')
        res_df.to_csv(save_path, index=False)

        logger.info(f"\n=========================================")
        logger.info(f"KERESÉS BEFEJEZVE!")
        logger.info(f"Összes talált alakzat: {len(res_df)}")
        logger.info(f"Eredmények mentve ide: {save_path}")
        logger.info(f"=========================================")
        print(res_df.head(10))
    else:
        logger.info("\nNem találtam alakzatot a megadott bizonyossági szint felett.")