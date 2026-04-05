"""
Configurations for Model Training
========================================
- Feature lists
"""
from dataclasses import dataclass, field
# ---- CONFIGURATIONS --------------------
@dataclass
class DataConfig:
  duckdb_path: str = "data/bgg_db.duckdb"

# ---- FEAATURES LISTS --------------------
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