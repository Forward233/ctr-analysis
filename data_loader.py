import pandas as pd
import os
from config import DATA_DIR


def load_data(data_path=None):
    path = data_path if data_path else os.path.join(DATA_DIR, "ecommerce_ctr.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return pd.read_csv(path)
