import logging
import sys
import subprocess

logger = logging.getLogger("MLLogger")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def has_nvidia_gpu():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False