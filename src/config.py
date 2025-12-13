import os
# Configuration settings as an example
ALLOWED_ASSETS = ['EURUSD', 'XAU']
N_BARS_LOOKBACK = 100 #for flag poles standardisation

# Download link
DOWNLOAD_LINK = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQAlEFc87da4SLpRVTCs81KwAS3DG4Ft8JPtUKQe9vV5eng?download=1"




BASE_DIR = '/home/bence/PycharmProjects/Melytanulas'
# LOCAL detektalas
LOCAL = os.path.exists(os.path.join(BASE_DIR, '.venv'))
if LOCAL:
    # --- LOKÁLIS ÚTVONALAK ---
    DATA_ROOT = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "notebook/output")
else:
    # --- DOCKER ÚTVONALAK ---
    DATA_ROOT = "/app/data"  # A mountolt volume (külső adat)
    OUTPUT_DIR = "/app/output"  # A konténer belső kimeneti mappája
# Mappák létrehozása, ha nem léteznének
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "baseline_lstm_best.pth")
LABEL_FILE = os.path.join(DATA_ROOT, "ground_truth_labels.csv")
# Paths
#DATA_ROOT = "../data/"
#OUTPUT_DIR = "./output"

#MODEL_SAVE_PATH = "./output/model.pth"
#LABEL_FILE = '../data/ground_truth_label.csv'


# Training hyperparameters
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 20

# --- HIPERPARAMÉTEREK ---

# ==============================
# 1. KÖZÖS PARAMÉTEREK
# ==============================
BATCH_SIZE = 32
SEQUENCE_LENGTH = 50
INPUT_SIZE = 4
LEARNING_RATE = 0.001
CLIP_VALUE = 1.0
DROPOUT = 0.2

# 2. BASELINE LSTM SPECIFIKUS
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# 3. HYBRID MODELL SPECIFIKUS
CNN_FILTERS = 32
D_MODEL = 64
N_HEADS = 4
DIM_FEEDFORWARD = 128
TRANSFORMER_LAYERS = 2
