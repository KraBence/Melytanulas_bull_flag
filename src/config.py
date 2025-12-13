import os
# Configuration settings as an example
ALLOWED_ASSETS = ['EURUSD', 'XAU']
N_BARS_LOOKBACK = 100 #for flag poles standardisation

# Download link
DOWNLOAD_LINK = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQAlEFc87da4SLpRVTCs81KwAS3DG4Ft8JPtUKQe9vV5eng?download=1"
INFERENCE_URL = "https://bmeedu-my.sharepoint.com/:x:/g/personal/gyires-toth_balint_vik_bme_hu/IQApaBhxGyYpR7opm3FFwereAeGD_3shTFw_0__izNdEy0M?download=1"

# --- HIPERPARAMÉTEREK ---

# 1. KÖZÖS PARAMÉTEREK
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 25
BATCH_SIZE = 64
SEQUENCE_LENGTH = 100
INPUT_SIZE = 4
LEARNING_RATE = 0.001
CLIP_VALUE = 1.0
DROPOUT = 0.22

# 2. BASELINE LSTM SPECIFIKUS
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# 3. HYBRID MODELL SPECIFIKUS
CNN_FILTERS = 64
D_MODEL = 128
N_HEADS = 4
DIM_FEEDFORWARD = 256
TRANSFORMER_LAYERS = 3


##### Some hardcoded parameters ####

BASE_DIR = '/home/bence/PycharmProjects/Melytanulas'
# LOCAL detektalas
LOCAL = os.path.exists(os.path.join(BASE_DIR, '.venv'))
if LOCAL:
    # --- LOKÁLIS ÚTVONALAK ---
    DATA_ROOT = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "notebook/output")
    LOG_DIR = os.path.join(BASE_DIR, "log")
else:
    # --- DOCKER ÚTVONALAK ---
    DATA_ROOT = "/app/data"  # A mountolt volume (külső adat)
    OUTPUT_DIR = "/app/output"  # A konténer belső kimeneti mappája
    LOG_DIR = "/app/log"

# Mappák létrehozása, ha nem léteznének
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "baseline_lstm_best.pth")
LABEL_FILE = os.path.join(DATA_ROOT, "ground_truth_labels.csv")
LOG_FILE = os.path.join(LOG_DIR, "run.log")




