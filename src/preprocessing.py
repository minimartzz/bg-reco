"""
Data Preprocessing Module
========================================
"""
import duckdb
import pandas as pd
from typing import Tuple
from config import DataConfig

def load_data(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
