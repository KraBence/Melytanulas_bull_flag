# Configuration settings as an example

N_BARS_LOOKBACK = 100 #for flag poles standardisation

# Training hyperparameters
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 10

# Paths
DATA_ROOT = "../data/bullflagdetector/bullflagdetector"
OUTPUT_DIR = "./output"

MODEL_SAVE_PATH = "/app/model.pth"

# --- HIPERPARAMÉTEREK ---
BATCH_SIZE = 32
SEQUENCE_LENGTH = 50
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 30
CLIP_VALUE = 1.0