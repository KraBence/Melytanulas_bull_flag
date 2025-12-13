import logging
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import config


def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class FlagDataset(Dataset):
    def __init__(self, metadata, csv_dir=config.OUTPUT_DIR, seq_len=config.SEQUENCE_LENGTH, augment=False):
        self.metadata = metadata
        self.csv_dir = csv_dir
        self.seq_len = seq_len
        self.augment = augment
        self.loaded_dfs = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # JAVÍTÁS: A helyes oszlopnév 'clean_csv_filename'
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

            tensor_segment = torch.tensor(segment_norm, dtype=torch.float32)

            if self.augment:
                tensor_segment += torch.randn_like(tensor_segment) * 0.01
                scale = 1.0 + (np.random.rand() - 0.5) * 0.1
                tensor_segment *= scale

            tensor_segment = tensor_segment.permute(1, 0).unsqueeze(0)
            tensor_resized = torch.nn.functional.interpolate(tensor_segment, size=self.seq_len, mode='linear',
                                                             align_corners=False)
            tensor_final = tensor_resized.squeeze(0).permute(1, 0)

            if torch.isnan(tensor_final).any(): return self._get_dummy()
            return tensor_final, torch.tensor(row['label_idx'], dtype=torch.long)
        except:
            return self._get_dummy()

    def _get_dummy(self):
        return torch.zeros((self.seq_len, config.INPUT_SIZE)), torch.tensor(0, dtype=torch.long)


# --- MODELLEK ---

class BaselineLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(config.HIDDEN_SIZE * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class HybridModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HybridModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, config.CNN_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.CNN_FILTERS), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(config.CNN_FILTERS, config.D_MODEL, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.D_MODEL), nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(config.D_MODEL, max_len=config.SEQUENCE_LENGTH)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL, nhead=config.N_HEADS, dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.TRANSFORMER_LAYERS)
        self.fc = nn.Sequential(
            nn.Linear(config.D_MODEL, 64), nn.ReLU(), nn.Dropout(config.DROPOUT),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        logits = self.fc(x)
        return logits