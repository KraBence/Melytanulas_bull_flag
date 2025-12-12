# Utility functions
# Common helper functions used across the project.
import logging
import sys

def setup_logger(name=__name__):
    """
    Sets up a logger that outputs to the console (stdout).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def load_config():
    pass


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --- GLOBÁLIS KONFIGURÁCIÓ (Közös) ---
# Dockerben:
OUTPUT_DIR = "/app/output"
# Lokális teszthez:
if not os.path.exists(OUTPUT_DIR): OUTPUT_DIR = "./output"

LABEL_FILE = os.path.join(OUTPUT_DIR, 'ground_truth_labels.csv')

# Modell és Adat Paraméterek
SEQUENCE_LENGTH = 50
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32  # Ez kell a loaderhez mindkét helyen


# --- DATASET OSZTÁLY ---
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


# --- MODELL OSZTÁLY ---
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