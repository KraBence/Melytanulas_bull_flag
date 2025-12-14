import os
import shutil
import glob
import sys

# Útvonal beállítása
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import config
from utils import setup_logger

logger = setup_logger()


def cleanup():
    logger.info("\n" + "=" * 50)
    logger.info(">>> 05 - TAKARÍTÁS (CLEANUP)")
    logger.info("=" * 50)

    data_dir = config.DATA_ROOT

    # 1. Inference letöltések törlése (Mappa)
    inf_dl_dir = os.path.join(data_dir, "inference_downloads")
    if os.path.exists(inf_dl_dir):
        try:
            shutil.rmtree(inf_dl_dir)
            logger.info(f"[TÖRÖLVE] Mappa: {inf_dl_dir}")
        except Exception as e:
            logger.error(f"Hiba a mappa törlésekor: {e}")

    # 2. Nyers és tisztított CSV-k törlése a data gyökérből
    # Minden CSV-t törlünk, KIVÉVE a ground_truth_labels.csv-t (az hasznos lehet)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    for f in csv_files:
        filename = os.path.basename(f)
        if "ground_truth_labels.csv" in filename:
            continue  # Ezt megtartjuk, mert ez az "eredménye" a preprocessingnek

        try:
            os.remove(f)
            logger.info(f"[TÖRÖLVE] Fájl: {filename}")
        except Exception as e:
            logger.error(f"Hiba a fájl törlésekor ({filename}): {e}")

    # 3. Egyéb szemetek (ZIP, tmp)
    temps = glob.glob(os.path.join(data_dir, "*.zip")) + glob.glob(os.path.join(data_dir, "*.tmp"))
    for t in temps:
        try:
            os.remove(t)
            logger.info(f"[TÖRÖLVE] Temp: {os.path.basename(t)}")
        except:
            pass

    logger.info("TAKARÍTÁS KÉSZ. A /data mappa tiszta.")


if __name__ == "__main__":
    cleanup()