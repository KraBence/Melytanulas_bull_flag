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

        # JAVÍTÁS: clean_csv_filename használata
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
        # Ha a seq_len változik (tuning közben), akkor dinamikusan kell kezelni,
        # de itt alapvetően a configot használjuk fallbacknek.
        # A __getitem__-ben a resize úgyis megoldja a méretet.
        return torch.zeros((self.seq_len, config.INPUT_SIZE)), torch.tensor(0, dtype=torch.long)


# ==========================================
# 3. BASELINE MODEL (LSTM)
# ==========================================
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


# ==========================================
# 4. HYBRID MODELL (CNN + Transformer)
# ==========================================
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
    def __init__(self, input_size, num_classes,
                 # Opcionális paraméterek (Hyperopt-hoz)
                 d_model=None, nhead=None, num_layers=None, dim_feedforward=None,
                 dropout=None, cnn_filters=None):
        super(HybridModel, self).__init__()

        # Ha nincs megadva, használja a configot
        D_MODEL = d_model if d_model else config.D_MODEL
        CNN_FILTERS = cnn_filters if cnn_filters else config.CNN_FILTERS
        N_HEADS = nhead if nhead else config.N_HEADS
        DIM_FF = dim_feedforward if dim_feedforward else config.DIM_FEEDFORWARD
        LAYERS = num_layers if num_layers else config.TRANSFORMER_LAYERS
        DROP = dropout if dropout else config.DROPOUT

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, CNN_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm1d(CNN_FILTERS), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(CNN_FILTERS, D_MODEL, kernel_size=3, padding=1),
            nn.BatchNorm1d(D_MODEL), nn.ReLU()
        )

        # Max len 5000 elég nagy tartaléknak
        self.pos_encoder = PositionalEncoding(D_MODEL, max_len=5000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=DIM_FF,
            dropout=DROP, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=LAYERS)

        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, 64),
            nn.GELU(),
            nn.Dropout(DROP),
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


# ==========================================
# 5. FOCAL LOSS (EZ HIÁNYZOTT!)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ... (Előző kódok: Imports, Logger, Dataset, Baseline, Hybrid, FocalLoss ...)

# ==========================================
# 6. ENSEMBLE MODEL (LSTM + CNN-Transformer)
# ==========================================
class EnsembleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnsembleModel, self).__init__()

        # --- ÁG 1: LSTM (Szekvenciális logika) ---
        # Ugyanaz a struktúra, mint a Baseline-nál, de FC réteg nélkül
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=True
        )
        # Az LSTM kimeneti mérete: Hidden * 2 (bidirectional)
        self.lstm_out_dim = config.HIDDEN_SIZE * 2

        # --- ÁG 2: HYBRID (Strukturális logika) ---
        # Ugyanaz a CNN+Transformer struktúra, mint a Hybridnél, FC réteg nélkül
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, config.CNN_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.CNN_FILTERS), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(config.CNN_FILTERS, config.D_MODEL, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.D_MODEL), nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(config.D_MODEL, max_len=5000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL, nhead=config.N_HEADS, dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.TRANSFORMER_LAYERS)
        # A Hybrid kimeneti mérete: D_MODEL
        self.hybrid_out_dim = config.D_MODEL

        # --- FUSION (Összefűzés) ---
        # A bemenet mérete a két ág összege
        combined_dim = self.lstm_out_dim + self.hybrid_out_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # --- 1. LSTM ÁG FUTTATÁSA ---
        # x: [Batch, Seq, Feat]
        lstm_out, _ = self.lstm(x)
        # Csak az utolsó lépést vesszük ki: [Batch, Hidden*2]
        lstm_feat = lstm_out[:, -1, :]

        # --- 2. HYBRID ÁG FUTTATÁSA ---
        # CNN-nek forgatni kell: [Batch, Feat, Seq]
        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in)

        # Transformernek vissza: [Batch, Seq, D_Model]
        trans_in = cnn_out.permute(0, 2, 1)
        trans_in = self.pos_encoder(trans_in)
        trans_out = self.transformer_encoder(trans_in)

        # Átlagolás (GAP): [Batch, D_Model]
        hybrid_feat = trans_out.mean(dim=1)

        # --- 3. ÖSSZEFŰZÉS (CONCATENATION) ---
        # [Batch, Hidden*2 + D_Model]
        combined = torch.cat((lstm_feat, hybrid_feat), dim=1)

        # --- 4. DÖNTÉS ---
        logits = self.fusion_head(combined)

        return logits