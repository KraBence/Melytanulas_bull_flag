# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
from utils import setup_logger
import config
N_BARS_LOOKBACK = config.N_BARS_LOOKBACK
DATA_ROOT = config.DATA_ROOT
OUTPUT_DIR = config.OUTPUT_DIR

import os
import json
import pandas as pd
import glob
import re

logger = setup_logger()

def robust_parse_ts(value):
    """Kezeli a Unix ms, Unix s és ISO string formátumokat is."""
    try:
        if pd.isna(value) or value == "": return pd.NaT
        if str(value).isdigit() or isinstance(value, (int, float)):
            val_int = int(value)
            if val_int > 30000000000: return pd.to_datetime(val_int, unit='ms')
            else: return pd.to_datetime(val_int, unit='s')
        return pd.to_datetime(value)
    except:
        return pd.NaT

def get_uniform_name(filename):
    if not filename: return "UNKNOWN"
    clean = filename
    if '-' in filename: parts = filename.split('-', 1); clean = parts[1] if len(parts)>1 else clean
    base = os.path.splitext(clean)[0]
    parts = base.split('_')
    if len(parts) >= 2:
        asset = parts[0]; tf = parts[1]
        if re.search(r'\d+\s*(minute|min|m|M)$', tf, re.IGNORECASE):
            tf = re.sub(r'(minute|min|m|M)$', 'min', tf, flags=re.IGNORECASE)
        elif re.search(r'\d+\s*(hour|h)$', tf, re.IGNORECASE):
            tf = re.sub(r'(hour|h)$', 'H', tf, flags=re.IGNORECASE)
        return f"{asset}_{tf}"
    return base

# ==========================================
# KONFLIKTUSKEZELÉS
# ==========================================
def filter_overlaps(df):
    """
    Kiszűri a duplikációkat és a súlyos átfedéseket.
    """
    if df.empty: return df

    logger("\n[SZŰRÉS] Átfedések és duplikációk vizsgálata...")
    original_count = len(df)

    # 1. Teljes duplikátumok törlése
    df['start_round'] = df['flag_start_ts'].dt.round('1min')
    df['end_round'] = df['flag_end_ts'].dt.round('1min')

    # Megtartjuk az elsőt
    df = df.drop_duplicates(subset=['raw_csv_filename', 'start_round', 'end_round', 'label'], keep='first')

    # 2. Konfliktusok keresése
    df = df.sort_values(by=['raw_csv_filename', 'flag_start_ts'])
    indices_to_drop = []

    for group_name, group in df.groupby('raw_csv_filename'):
        group = group.reset_index()
        for i in range(len(group) - 1):
            curr = group.iloc[i]
            next_row = group.iloc[i+1]

            # Átfedés vizsgálata
            if curr['flag_end_ts'] > next_row['flag_start_ts']:
                intersection = min(curr['flag_end_ts'], next_row['flag_end_ts']) - next_row['flag_start_ts']
                union = max(curr['flag_end_ts'], next_row['flag_end_ts']) - curr['flag_start_ts']

                if union.total_seconds() > 0:
                    overlap_ratio = intersection.total_seconds() / union.total_seconds()
                else:
                    overlap_ratio = 0

                if overlap_ratio > 0.5:
                    if curr['label'] != next_row['label']:
                        logger(f"    [KONFLIKTUS] {group_name}: Megtartva: {curr['label']} | Eldobva: {next_row['label']}")
                    indices_to_drop.append(next_row['index'])

    if indices_to_drop:
        df = df.drop(indices_to_drop)
        logger(f"    -> Eltávolítva {len(indices_to_drop)} db konfliktusos sor.")

    df = df.drop(columns=['start_round', 'end_round'])
    logger(f"    -> Maradt: {len(df)} (Eredeti: {original_count})")
    return df

# ==========================================
# 1. FÁZIS: CSV ÖSSZEFŰZÉS
# ==========================================
def merge_and_save_csvs(data_dir, output_dir):
    logger("\n[1. FÁZIS] CSV fájlok keresése és összefűzése...")
    all_csvs = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    grouped_dfs = {}

    for csv_path in all_csvs:
        if output_dir in csv_path: continue
        group_name = get_uniform_name(os.path.basename(csv_path))
        try:
            with open(csv_path, 'r') as f: line = f.readline()
            sep = ';' if ';' in line else ','
            df = pd.read_csv(csv_path, sep=sep)
            df.columns = df.columns.str.lower().str.strip()

            t_col = next((c for c in df.columns if 'time' in c or 'date' in c), df.columns[0])
            vals = pd.to_numeric(df[t_col], errors='coerce')
            if vals.notna().mean() > 0.8:
                if vals.max() > 300000000000: df['dt'] = pd.to_datetime(vals, unit='ms')
                else: df['dt'] = pd.to_datetime(vals, unit='s')
            else:
                df['dt'] = pd.to_datetime(df[t_col], errors='coerce')

            df = df.dropna(subset=['dt']).set_index('dt')
            cols = [c for c in df.columns if c in ['open', 'high', 'low', 'close', 'volume']]
            df = df[cols]

            if group_name not in grouped_dfs: grouped_dfs[group_name] = []
            grouped_dfs[group_name].append(df)
        except: pass

    saved_map = {}
    os.makedirs(output_dir, exist_ok=True)
    for group_name, df_list in grouped_dfs.items():
        if not df_list: continue
        full_df = pd.concat(df_list).sort_index()
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        save_path = os.path.join(output_dir, f"merged_{group_name}.csv")
        full_df.to_csv(save_path)
        saved_map[group_name] = save_path
        logger(f"    -> Mentve: merged_{group_name}.csv ({len(full_df)} sor)")
    return saved_map

# ==========================================
# 2. FÁZIS: CÍMKE FELDOLGOZÁS
# ==========================================
def find_best_pole(label_row, ohlcv_df):
    try:
        anchor_idx = ohlcv_df.index.get_indexer([label_row['flag_start_ts']], method='nearest')[0]
        anchor_bar = ohlcv_df.iloc[anchor_idx]
        if abs((anchor_bar.name - label_row['flag_start_ts']).total_seconds()) > 14400: return None
    except: return None

    best_ts = None; max_slope = -float('inf')
    for i in range(1, N_BARS_LOOKBACK + 1):
        cand_idx = anchor_idx - i
        if cand_idx < 0: break
        cand_bar = ohlcv_df.iloc[cand_idx]
        p_change = 0.0
        if label_row['pattern_type'] == "BULL_FLAG": p_change = anchor_bar['high'] - cand_bar['low']
        elif label_row['pattern_type'] == "BEAR_FLAG": p_change = cand_bar['high'] - anchor_bar['low']

        if p_change > 0:
            slope = p_change / i
            if slope > max_slope: max_slope = slope; best_ts = cand_bar.name
    return best_ts

def process_labels(data_dir, output_dir, merged_files_map):
    logger("\n[2. FÁZIS] Címkék feldolgozása...")
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    final_dataset = []
    loaded_dfs = {}

    for jpath in json_files:
        if 'sample' in jpath or 'consensus' in jpath: continue
        try:
            with open(jpath, 'r') as f: content = json.load(f)
            if isinstance(content, dict): content = [content]
            for task in content:
                ls_filename = task.get('file_upload')
                if not ls_filename: continue
                uniform_name = get_uniform_name(ls_filename)

                merged_path = merged_files_map.get(uniform_name)
                if not merged_path:
                    for k, v in merged_files_map.items():
                        if uniform_name in k: merged_path = v; uniform_name = k; break
                if not merged_path: continue

                if uniform_name not in loaded_dfs:
                    loaded_dfs[uniform_name] = pd.read_csv(merged_path, index_col=0, parse_dates=True)
                ohlcv_df = loaded_dfs[uniform_name]

                for ann in task.get('annotations', []):
                    for res in ann.get('result', []):
                        val = res.get('value', {})
                        if val.get('timeserieslabels'):
                            lbl = val['timeserieslabels'][0]

                            # Típus meghatározása
                            p_type = "BULL_FLAG" if "Bullish" in lbl else "BEAR_FLAG" if "Bearish" in lbl else "UNKNOWN"

                            # --- ÚJ: TREND CÍMKE (BULL / BEAR) ---
                            trend = "UNKNOWN"
                            if p_type == "BULL_FLAG": trend = "BULL"
                            elif p_type == "BEAR_FLAG": trend = "BEAR"

                            start_ts = robust_parse_ts(val['start'])
                            end_ts = robust_parse_ts(val['end'])
                            if pd.isna(start_ts): continue

                            temp_row = {'flag_start_ts': start_ts, 'pattern_type': p_type}
                            pole_ts = find_best_pole(temp_row, ohlcv_df)

                            if pole_ts:
                                final_dataset.append({
                                    "full_filename": ls_filename,
                                    "raw_csv_filename": uniform_name,
                                    "merged_csv_path": f"merged_{uniform_name}.csv",
                                    "label": lbl,          # Pl: "Bullish Wedge"
                                    "trend_label": trend,  # Pl: "BULL"
                                    "flag_start_ts": start_ts,
                                    "flag_end_ts": end_ts,
                                    "pole_start_ts": pole_ts,
                                    "pattern_type": p_type
                                })
        except: pass

    df_result = pd.DataFrame(final_dataset)
    df_result = filter_overlaps(df_result)
    return df_result

if __name__ == "__main__":
    merged_map = merge_and_save_csvs(DATA_ROOT, OUTPUT_DIR)
    if merged_map:
        df_final = process_labels(DATA_ROOT, OUTPUT_DIR, merged_map)
        out_file = os.path.join(OUTPUT_DIR, "ground_truth_labels.csv")
        df_final.to_csv(out_file, index=False)
        logger(f"\n[KÉSZ] Sikeresen generálva: {len(df_final)} sor.")
        if not df_final.empty:
            logger(df_final[['label', 'trend_label']].head())
    else:
        logger("HIBA: Nem sikerült CSV-ket feldolgozni.")