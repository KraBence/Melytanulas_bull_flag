import logging
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Import configuration
import config


# ==========================================
# 1. LOGGER SETUP
# ==========================================
def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# ==========================================
# 2. DATASET CLASS
# ==========================================
class FlagDataset(Dataset):
    def __init__(self, metadata, csv_dir=config.OUTPUT_DIR, seq_len=config.SEQUENCE_LENGTH):
        self.metadata = metadata
        self.csv_dir = csv_dir
        self.seq_len = seq_len
        self.loaded_dfs = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        csv_filename = row['clean_csv_filename']
        csv_path = os.path.join(self.csv_dir, csv_filename)

        if csv_filename not in self.loaded_dfs:
            if os.path.exists(csv_path):
                try:
                    self.loaded_dfs[csv_filename] = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                except:
                    self.loaded_dfs[csv_filename] = None
            else:
                self.loaded_dfs[csv_filename] = None

        df = self.loaded_dfs[csv_filename]
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
        # Access INPUT_SIZE via config
        return torch.zeros((self.seq_len, config.INPUT_SIZE)), torch.tensor(0, dtype=torch.long)


# ==========================================
# 3. BASELINE MODEL (LSTM)
# ==========================================
class BaselineLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BaselineLSTM, self).__init__()

        # JAVÍTÁS: A config-ból vesszük az értékeket, nem fix számokat írunk!
        # Config: HIDDEN_SIZE = 64, NUM_LAYERS = 2, DROPOUT = 0.2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.HIDDEN_SIZE,  # 64
            num_layers=config.NUM_LAYERS,  # 2
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,  # 0.2
            bidirectional=True
        )

        # Mivel bidirectional, a kimenet mérete: HIDDEN_SIZE * 2
        self.fc = nn.Linear(config.HIDDEN_SIZE * 2, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)

        # Az utolsó időpillanat kimenete (mindkét irányból)
        last_step_out = out[:, -1, :]

        logits = self.fc(last_step_out)
        return logits


# ==========================================
# 4. HYBRID MODELL (CNN + Transformer)
# ==========================================
class HybridModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HybridModel, self).__init__()

        # 1. CNN Feature Extractor (Alakzatok felismerése)
        # Bemenet: (Batch, Input_Size, Seq_Len) -> Ezért majd forgatni kell
        self.cnn = nn.Sequential(
            # Első konvolúció: Alapvető élek, vonalak
            nn.Conv1d(in_channels=input_size, out_channels=config.CNN_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.CNN_FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Felezi a hosszt (50 -> 25)

            # Második konvolúció: Komplexebb formák + Illesztés a Transformerhez (D_MODEL)
            nn.Conv1d(in_channels=config.CNN_FILTERS, out_channels=config.D_MODEL, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.D_MODEL),
            nn.ReLU()
        )

        # 2. Transformer Encoder (Időbeli összefüggések)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.TRANSFORMER_LAYERS)

        # 3. Osztályozó fej
        self.fc = nn.Sequential(
            nn.Linear(config.D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x bemenet: [batch, seq_len, input_size] (pl. 32, 50, 4)

        # A CNN [batch, channels, length] formátumot vár -> Permute
        x = x.permute(0, 2, 1)

        # CNN futtatása
        x = self.cnn(x)
        # Kimenet: [batch, d_model, seq_len/2]

        # Visszaforgatjuk a Transformernek: [batch, length, d_model]
        x = x.permute(0, 2, 1)

        # Transformer futtatása
        x = self.transformer_encoder(x)

        # Global Average Pooling (GAP)
        # Az idődimenziót átlagoljuk, hogy egyetlen vektort kapjunk a sorozathoz
        x = x.mean(dim=1)

        # Osztályozás
        logits = self.fc(x)
        return logits