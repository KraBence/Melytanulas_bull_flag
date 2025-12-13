import os
import json
import pandas as pd
import glob
import re
import requests
import zipfile
import config
from utils import setup_logger

# Initialize Logger
logger = setup_logger()

# --- CONFIGURATION ---
N_BARS_LOOKBACK = config.N_BARS_LOOKBACK
ALLOWED_ASSETS = config.ALLOWED_ASSETS
DATA_ROOT = config.DATA_ROOT

OUTPUT_DIR = DATA_ROOT

# 0. DATA DOWNLOADER
def download_and_setup_data(url=config.DOWNLOAD_LINK, output_dir=DATA_ROOT):
    """
    Downloads the ZIP file from the specified URL and extracts it to the target directory.
    """
    # 1. Create target directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Directory created: {output_dir}")

    # Temporary filename for download
    temp_zip = "temp_dataset_download.zip"

    logger.info(f"Starting download from: {url} ...")

    try:
        # 2. Download with streaming
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error if link is unreachable

        total_size = 0
        with open(temp_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)

        size_mb = total_size / (1024 * 1024)
        logger.info(f"Download complete! Size: {size_mb:.2f} MB")
        logger.info("Unzipping in progress...")

        # 3. Extract to target directory
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        logger.info(f"Successfully extracted to: {output_dir}")

    except Exception as e:
        logger.error(f"Error occurred during download/extraction: {e}")

    finally:
        # 4. Cleanup: remove temporary zip
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
            logger.info("Temporary files deleted.")


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def robust_parse_ts(value):
    """Handles Unix ms, Unix s, and ISO string formats."""
    try:
        if pd.isna(value) or value == "": return pd.NaT
        if str(value).replace('.', '', 1).isdigit():  # Handle float strings
            val_float = float(value)
            if val_float > 30000000000:
                return pd.to_datetime(val_float, unit='ms')
            else:
                return pd.to_datetime(val_float, unit='s')
        return pd.to_datetime(value)
    except:
        return pd.NaT


def get_uniform_name(filename):
    if not filename: return "UNKNOWN"
    clean = filename
    if '-' in filename: parts = filename.split('-', 1); clean = parts[1] if len(parts) > 1 else clean
    base = os.path.splitext(clean)[0]
    parts = base.split('_')
    if len(parts) >= 2:
        asset = parts[0];
        tf = parts[1]
        if re.search(r'\d+\s*(minute|min|m|M)$', tf, re.IGNORECASE):
            tf = re.sub(r'(minute|min|m|M)$', 'min', tf, flags=re.IGNORECASE)
        elif re.search(r'\d+\s*(hour|h)$', tf, re.IGNORECASE):
            tf = re.sub(r'(hour|h)$', 'H', tf, flags=re.IGNORECASE)
        return f"{asset}_{tf}"
    return base


def is_asset_allowed(filename):
    fname_upper = filename.upper()
    for asset in ALLOWED_ASSETS:
        if asset in fname_upper:
            return True
    return False


# ==========================================
# 2. OVERLAP FILTERING
# ==========================================
def filter_overlaps(df):
    if df.empty: return df

    logger.info("\n[FILTERING] Cleaning overlaps...")
    original_count = len(df)

    df['start_round'] = df['flag_start_ts'].dt.round('1min')
    df['end_round'] = df['flag_end_ts'].dt.round('1min')

    df = df.drop_duplicates(subset=['clean_csv_filename', 'start_round', 'end_round', 'label'], keep='first')

    df = df.sort_values(by=['clean_csv_filename', 'flag_start_ts'])
    indices_to_drop = []

    for filename, group in df.groupby('clean_csv_filename'):
        group = group.reset_index()
        for i in range(len(group) - 1):
            curr = group.iloc[i]
            next_row = group.iloc[i + 1]

            if curr['flag_end_ts'] > next_row['flag_start_ts']:
                intersection = min(curr['flag_end_ts'], next_row['flag_end_ts']) - next_row['flag_start_ts']
                union = max(curr['flag_end_ts'], next_row['flag_end_ts']) - curr['flag_start_ts']

                overlap_ratio = 0
                if union.total_seconds() > 0:
                    overlap_ratio = intersection.total_seconds() / union.total_seconds()

                if overlap_ratio > 0.5:
                    indices_to_drop.append(next_row['index'])

    if indices_to_drop:
        df = df.drop(indices_to_drop)

    df = df.drop(columns=['start_round', 'end_round'])
    logger.info(f"    -> Remaining labels: {len(df)} (Original: {original_count})")
    return df


# ==========================================
# 3. PROCESS AND SAVE CSVS (NO MERGE)
# ==========================================
def process_and_save_csvs(data_dir, output_dir):
    logger.info(f"\n[PHASE 1] Searching and cleaning CSV files...")

    all_csvs = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    kept_files_map = {}

    created_filenames = set()

    for csv_path in all_csvs:
        filename = os.path.basename(csv_path)

        if not is_asset_allowed(filename):
            continue

        if filename.startswith("clean_") or "ground_truth_labels" in filename:
            continue

        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                line = f.readline()
            sep = ';' if ';' in line else ','

            df = pd.read_csv(csv_path, sep=sep)
            df.columns = df.columns.str.lower().str.strip()

            t_col = next((c for c in df.columns if 'time' in c or 'date' in c), None)
            if t_col is None: t_col = df.columns[0]

            vals = pd.to_numeric(df[t_col], errors='coerce')
            if vals.notna().mean() > 0.8:
                if vals.max() > 300000000000:
                    df['dt'] = pd.to_datetime(vals, unit='ms')
                else:
                    df['dt'] = pd.to_datetime(vals, unit='s')
            else:
                df['dt'] = pd.to_datetime(df[t_col], errors='coerce')

            df = df.dropna(subset=['dt']).set_index('dt').sort_index()

            required_cols = ['open', 'high', 'low', 'close']
            available_cols = [c for c in df.columns if c in required_cols or c == 'volume']

            if not all(col in available_cols for col in required_cols):
                logger.warning(f"    [SKIP] Missing OHLC columns: {filename}")
                continue

            df = df[available_cols]
            df = df[~df.index.duplicated(keep='first')]

            # Unique filename generation
            base_clean_name = f"clean_{filename}"
            final_save_name = base_clean_name
            save_path = os.path.join(output_dir, final_save_name)

            counter = 1
            while os.path.exists(save_path) or final_save_name in created_filenames:
                name_part, ext_part = os.path.splitext(base_clean_name)
                final_save_name = f"{name_part}_dup{counter}{ext_part}"
                save_path = os.path.join(output_dir, final_save_name)
                counter += 1

            df.to_csv(save_path)
            created_filenames.add(final_save_name)

            # Map original filename to the cleaned, saved filename
            kept_files_map[filename] = save_path

            if counter > 1:
                logger.info(f"    -> Saved (Renamed): {final_save_name} ({len(df)} rows)")
            else:
                logger.info(f"    -> Saved: {final_save_name} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"    [ERROR] {filename}: {e}")

    return kept_files_map


# ==========================================
# 4. LABEL PROCESSING
# ==========================================
def find_best_pole(label_row, ohlcv_df):
    try:
        # Nearest anchor point
        idx_locs = ohlcv_df.index.get_indexer([label_row['flag_start_ts']], method='nearest')
        anchor_idx = idx_locs[0]
        anchor_bar = ohlcv_df.iloc[anchor_idx]

        # Check time difference (max 4 hours)
        if abs((anchor_bar.name - label_row['flag_start_ts']).total_seconds()) > 14400:
            return None
    except:
        return None

    best_ts = None
    max_slope = -float('inf')

    # Look back N bars
    for i in range(1, N_BARS_LOOKBACK + 1):
        cand_idx = anchor_idx - i
        if cand_idx < 0: break
        cand_bar = ohlcv_df.iloc[cand_idx]

        p_change = 0.0
        if label_row['pattern_type'] == "BULL_FLAG":
            p_change = anchor_bar['high'] - cand_bar['low']
        elif label_row['pattern_type'] == "BEAR_FLAG":
            p_change = cand_bar['high'] - anchor_bar['low']

        if p_change > 0:
            slope = p_change / i
            if slope > max_slope:
                max_slope = slope
                best_ts = cand_bar.name
    return best_ts


def process_labels(data_dir, output_dir, kept_files_map):
    logger.info("\n[PHASE 2] Processing labels...")
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    final_dataset = []
    loaded_dfs = {}

    for jpath in json_files:
        if 'sample' in jpath or 'consensus' in jpath: continue
        try:
            with open(jpath, 'r') as f:
                content = json.load(f)
            if isinstance(content, dict): content = [content]

            for task in content:
                original_filename = task.get('file_upload')
                original_filename_base = os.path.basename(original_filename) if original_filename else None

                if not original_filename_base: continue

                # Look up cleaned file path
                clean_path = kept_files_map.get(original_filename_base)

                if not clean_path:
                    # Try partial match if direct match fails
                    found = False
                    for k, v in kept_files_map.items():
                        if k in original_filename_base or original_filename_base in k:
                            clean_path = v
                            found = True
                            break
                    if not found: continue

                clean_filename = os.path.basename(clean_path)
                if clean_filename not in loaded_dfs:
                    loaded_dfs[clean_filename] = pd.read_csv(clean_path, index_col=0, parse_dates=True)
                ohlcv_df = loaded_dfs[clean_filename]

                for ann in task.get('annotations', []):
                    for res in ann.get('result', []):
                        val = res.get('value', {})
                        if val.get('timeserieslabels'):
                            lbl = val['timeserieslabels'][0]
                            p_type = "BULL_FLAG" if "Bullish" in lbl else "BEAR_FLAG" if "Bearish" in lbl else "UNKNOWN"
                            trend = "BULL" if p_type == "BULL_FLAG" else "BEAR" if p_type == "BEAR_FLAG" else "UNKNOWN"

                            start_ts = robust_parse_ts(val['start'])
                            end_ts = robust_parse_ts(val['end'])
                            if pd.isna(start_ts): continue

                            temp_row = {'flag_start_ts': start_ts, 'pattern_type': p_type}
                            pole_ts = find_best_pole(temp_row, ohlcv_df)

                            if pole_ts:
                                final_dataset.append({
                                    "original_filename": original_filename_base,
                                    "clean_csv_filename": clean_filename,
                                    "label": lbl,
                                    "trend_label": trend,
                                    "flag_start_ts": start_ts,
                                    "flag_end_ts": end_ts,
                                    "pole_start_ts": pole_ts,
                                    "pattern_type": p_type
                                })
        except Exception as e:
            logger.error(f"Error processing JSON ({jpath}): {e}")

    df_result = pd.DataFrame(final_dataset)
    df_result = filter_overlaps(df_result)
    return df_result


# ==========================================
# 5. CLEANUP
# ==========================================
def cleanup_data_folder(data_dir, kept_files_map):
    logger.info("\n[PHASE 3] Cleanup...")

    keep_set = set([os.path.abspath(p) for p in kept_files_map.values()])
    keep_set.add(os.path.abspath(os.path.join(data_dir, "ground_truth_labels.csv")))

    all_files = glob.glob(os.path.join(data_dir, "**/*"), recursive=True)
    deleted_count = 0

    for f in all_files:
        if os.path.isdir(f): continue
        if os.path.abspath(f) not in keep_set:
            try:
                os.remove(f)
                deleted_count += 1
            except:
                pass

    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except:
                pass

    logger.info(f"    -> Deleted {deleted_count} files.")
    logger.info(f"    -> Kept files: {len(keep_set) - 1} data files + labels.")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    download_and_setup_data()
    kept_files_map = process_and_save_csvs(DATA_ROOT, OUTPUT_DIR)

    if kept_files_map:
        df_final = process_labels(DATA_ROOT, OUTPUT_DIR, kept_files_map)

        out_file = os.path.join(OUTPUT_DIR, "ground_truth_labels.csv")
        df_final.to_csv(out_file, index=False)

        logger.info(f"\n[DONE] Ground Truth generated: {len(df_final)} rows.")
        if not df_final.empty:
            logger.info(str(df_final[['label', 'trend_label']].head()))

        cleanup_data_folder(DATA_ROOT, kept_files_map)

    else:
        logger.error("ERROR: No relevant files (XAU/EURUSD) could be processed.")