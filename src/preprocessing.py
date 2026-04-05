"""
Data Preprocessing Module
========================================
Contains functions for loading and preprocessing raw games, comments and records data

COMPONENTS:
- load_data: Loads data from DuckDB and Supabase
- build_tag_vocabularies: Builds vocabularies for multi-hot encoding of tag features
- multi_hot_encode: Encodes tag lists into multi-hot vectors
- fit_numeric_scaler: Fits a standard scaler on numeric features
- scale_numeric_features: Scales numeric features using the fitted scaler
"""
import os
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from supabase import create_client, Client
from typing import Tuple, Dict, Set, List
from config import DataConfig, GAME_TAG_COLUMNS, GAME_NUMERIC_FEATURES
from sklearn.preprocessing import StandardScaler

def load_data(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Loads data from both DuckDB (comments, games) and Supabase (records)

  Args:
    cfg (DataConfig): Data configuration

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Records, Games and Comments
  """
  # Get the records from Supabase
  try:
    supabase: Client = create_client(cfg.supa_url, cfg.supa_key)
    response = (
      supabase.table(cfg.records_table)
      .select("*")
      .execute()
    )
    records = pd.DataFrame(response.data)
  except Exception as e:
    print(f"Error fetching data from Supabase: {e}")
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Returns empty DataFrames
  
  # Load games and comments from DuckDB
  try:
    base_dir = Path(cfg.duckdb_file).resolve().parent.parent
    duckdb_path = base_dir / "data" / cfg.duckdb_file
    con = duckdb.connect(duckdb_path)
    games = con.execute("SELECT * FROM bgg.games").fetchdf()
    comments = con.execute("SELECT * FROM bgg.comments").fetchdf()
  except Exception as e:
    print(f"Error fetching data from DuckDB: {e}")
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Returns empty DataFrames

  return records, games, comments

# ---- BUILDING VOCABULARIES --------------------
def build_tag_vocabularies(
  games: pd.DataFrame,
  tag_columns: list = GAME_TAG_COLUMNS,
) -> Dict[str, Dict[str, int]]:
  """
  Builds a token -> index mapping for each tag column to create multi-hot vectors
  For each tag list column, a numerical index is assigned to each tag

  Args:
    games (pd.DataFrame): Games dataframe
    tag_columns (list, optional): List of tag columns to use. Defaults to GAME_TAG_COLUMNS.

  Returns:
    Dict[str, Dict[str, int]]: Mapping every tag to a corresponding index
  """
  vocabs: Dict[str, Dict[str, int]] = {}
  for col in tag_columns:
    all_tags = Set[str] = set()
    for tags in games[col]:
      all_tags.update(tags)
    vocabs[col] = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
  return vocabs

def multi_hot_encode(tags: List[str], vocab: Dict[str, int]) -> np.ndarray:
  """
  Encodes a list of tags into a multi-hot vector based on the provided vocabulary.
  Used to encode a games tags into a fixed-length binary vector

  Args:
    tags (List[str]): List of tags to encode
    vocab (Dict[str, int]): Vocabulary mapping tags to indices

  Returns:
    np.ndarray: Multi-hot encoded vector
  """
  vector = np.zeros(len(vocab), dtype=np.float32)
  for tag in tags:
    if tag in vocab:
      vector[vocab[tag]] = 1.0
  return vector

# ---- NORMALIZATION --------------------
def fit_numeric_scaler(games: pd.DataFrame) -> StandardScaler:
  """
  Fits a standard scaler on numeric columns while filling missing fields with default 0.

  Args:
      games (pd.DataFrame): Games dataframe

  Returns:
      StandardScaler: Fitted scaler
  """
  scaler = StandardScaler()
  scaler.fit(games[GAME_NUMERIC_FEATURES].fillna(0).values)
  return scaler

def scale_numeric_features(games: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
  """Scales numeric features using the provided scaler, filling missing values with 0."""
  return scaler.transform(games[GAME_NUMERIC_FEATURES].fillna(0).values).astype(np.float32)