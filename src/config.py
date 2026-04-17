"""
Configurations for Model Training
========================================
- Feature lists
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List

load_dotenv()

# ========================================
# CONFIGURATIONS
# ========================================
@dataclass
class DataConfig:
  duckdb_file: str = "bgg_db.duckdb"
  supa_url: str = os.environ.get("SUPABASE_URL")
  supa_key: str = os.environ.get("SUPABASE_KEY")
  records_table: str = "comp_game_log"

@dataclass
class EmbeddingConfig:
  model_name: str = "all-MiniLM-L6-v2"
  embedding_dim: int = 128
  use_sentence_transformers: bool = True
  max_comments_per_game: int = 50
  batch_size: int = 32

@dataclass
class ModelConfig:
  tower_output_dim: int = 128
  # Hidden layers for each tower
  game_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
  user_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
  # Reranker layers
  reranker_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
  dropout: float = 0.2
  # Temperature for InfoNCE loss
  temperature: float = 0.07

@dataclass
class TrainingConfig:
  epochs: int = 20
  batch_size: int = 128
  learning_rate: float = 1e-3
  weight_decay: float = 1e-5 
  # Negative sampling ratio for contrastive training
  num_negatives: int = 10
  # Train/ Val split
  val_fraction: float = 0.2
  # Reranker training
  reranker_epochs: int = 70
  reranker_lr: float = 5e-4
  # Number of candidates the two-tower model passes to reranker
  retrieval_top_k = 50


# ========================================
# FEATURES LISTS
# ========================================
# Numeric features
GAME_NUMERIC_FEATURES = [
  "min_players",
  "max_players",
  "suggested_players",
  "playing_time",
  "min_age",
  "complexity",
  "rating",
  "num_ratings",
  "year_published"
]

# Tag features
GAME_TAG_COLUMNS = [
  "categories",
  "mechanics",
  "families",
  "artists",
  "designers"
]

GAME_TAG_IDS_COLUMNS = [
  "category_ids",
  "mechanic_ids",
  "family_ids",
  "artist_ids",
  "designer_ids"
]