"""
Configurations for Model Training
========================================
- Feature lists
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()

# ---- CONFIGURATIONS --------------------
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

# ---- FEATURES LISTS --------------------
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