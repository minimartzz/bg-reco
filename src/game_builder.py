"""
Game Profile Builder
========================================
Creates a "profile" for each game by combining various features about one.

1. Numeric features: (e.g game length, complexity, number of players, etc.) are normalised
and stored in a dense vector. These captures the games "physical" characteristics.

2. Tag features: (e.g categories, mechanics, families) are multi-hot encoded into a sparse vector.
These captures the games "conceptual" characteristics and are always fixed in lenght so storing as
sparse vectors allow the model to learn sharp feature interactions when comparing a user's profile.

3. Text features: (e.g game description, comments) are captured using embeddings. Each are processed separately
to retain their different signals. Game descriptions capture the factual descriptions about the game, while
comments capture the subjective opinions of players. Separately embedding then pooling allows tower to learn
the relative importance of objective vs subjective signals

All 3 sub-vectors are concatenated to create a single profile vector that captures a game. The game tower then
projects this into a the shared embedding space allowing downstream model to weight each modality differently
"""
import numpy as np
import pandas as pd
from typing import Protocol, List, Dict, Optional
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from config import EmbeddingConfig, GAME_TAG_COLUMNS, GAME_NUMERIC_FEATURES
from preprocessing import multi_hot_encode

# ========================================
# TEXT ENCODERS
# ========================================
class TextEncoder(Protocol):
    """Encode a list of strings into dense vectors."""
    def encode(self, texts: List[str]) -> np.ndarray: ...

    @property
    def dim(self) -> int: ...

class TfidfSvdEncoder:
  """
  TF-IDF + Truncated SVD encoding.

  Offline fallback. Primary production use case will be sentence transformer
  """
  def __init__(self, dim: int = 128):
    self._dim = dim
    self.vectorizer = TfidfVectorizer(
      max_features=5000,
      stop_words='english',
      ngram_range=(1, 2),
      sublinear_tf=True # Sublinear term frequency scaling to reduce impact of high-frequency terms
    )
    self.svd = TruncatedSVD(n_components=dim, random_state=42)
    self._fitted = False
  
  @property
  def dim(self) -> int:
    return self._dim
  
  def fit(self, corpus: List[str]) -> "TfidfSvdEncoder":
    """Fits corpus of descriptions + comments"""
    tfidf = self.vectorizer.fit_transform(corpus)
    actual_components = min(self._dim, tfidf.shape[1], tfidf.shape[0])
    if actual_components < self._dim:
      self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
      self._dim = actual_components
    self.svd.fit(tfidf)
    self._fitted = True
    return self
  
  def encode(self, texts: List[str]) -> np.ndarray:
    """For prediction on next text: Transforms text to (n_text, dim) dense matrix"""
    tfidf = self.vectorizer.transform(texts)
    return self.svd.transform(tfidf).astype(np.float32)
    
class SentenceTransformerEncoder:
  def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
    self.model = SentenceTransformer(model_name)
    self._dim = self.model.get_sentence_embedding_dimension()
    self.batch_size = batch_size
  
  @property
  def dim(self) -> int:
    return self._dim
  
  def encode(self, texts: List[str]) -> np.ndarray:
    return self.model.encode(
      texts,
      batch_size=self.batch_size,
      show_progress_bar=True,
    ).astype(np.float32)

def build_encoder(
  descriptions: List[str],
  comments: List[str],
  cfg: EmbeddingConfig
):
  if cfg.use_sentence_transformers:
    try:
      enc = SentenceTransformerEncoder(cfg.model_name, cfg.batch_size)
      cfg.embedding_dim = enc.dim
      return enc
    except Exception as e:
      print(f"[GAME BUILDER] Error initializing the sentence transformer encoder: {e}")
  
  corpus = [t for t in descriptions + comments if t.strip()]
  enc = TfidfSvdEncoder(cfg.embedding_dim)
  enc.fit(corpus)
  return enc


# ========================================
# TEXT EMBEDDING
# ========================================
def embed_descriptions(games: pd.DataFrame, encoder: TextEncoder) -> Dict[int, np.ndarray]:
  """Embed each game's description and label them by game ID"""
  ids = games['id'].tolist()
  descriptions = games['description'].fillna("").tolist()
  embeddings = encoder.encode(descriptions)
  return dict(zip(ids, embeddings))

def embed_comments(
  comments: pd.DataFrame,
  encoder: TextEncoder,
  cfg: EmbeddingConfig
) -> Dict[int, np.ndarray]:
  grouped = comments.groupby("bgg_id")
  results: Dict[int, np.ndarray] = {}

  for game_id, group in grouped:
    group = group.head(cfg.max_comments_per_game)
    comments = group['comment'].fillna("").tolist()
    ratings = group['rating'].fillna(group['rating'].mean()).values.astype(np.float32)

    if not comments:
      results[int(game_id)] = np.zeros(encoder.dim, dtype=np.float32)
      continue

    embs = encoder.encode(comments)

    # Weight comment embeddings by their ratings
    weights = ratings / (ratings.sum() + 1e-8)
    weighted_embs = (embs * weights[:, None]).sum(axis=0)
    results[int(game_id)] = weighted_embs
  
  return results


# ========================================
# BUILD GAME PROFILES
# ========================================
def build_game_profiles(
  games: pd.DataFrame,
  comments: pd.DataFrame,
  tag_vocabs: Dict[str, Dict[str, int]],
  numeric_features: np.ndarray,
  cfg: EmbeddingConfig = EmbeddingConfig(),
  encoder: Optional[TextEncoder] = None
) -> tuple[Dict[int, np.ndarray], TextEncoder]:
  if encoder is None:
    all_desc = games['description'].fillna("").tolist()
    all_comments = comments['comment'].fillna("").tolist()
    encoder = build_encoder(all_desc, all_comments, cfg)
    cfg.embedding_dim = encoder.dim
  
  desc_embs = embed_descriptions(games, encoder)
  comment_embs = embed_comments(comments, encoder, cfg)

  profiles = Dict[int, np.ndarray] = {}

  for row_idx, row in games.iterrows():
    game_id = int(row['id'])

    # Text-based features
    desc_emb = desc_embs.get(game_id, np.zeros(encoder.dim, dtype=np.float32))
    com_emb = comment_embs.get(game_id, np.zeros(encoder.dim, dtype=np.float32))

    # Tag-based features
    tag_vecs = []
    for col in GAME_TAG_COLUMNS:
      vocab = tag_vocabs[col]
      tags = row.get(col, [])
      if not isinstance(tags, list):
        tags = []
      tag_vecs.append(multi_hot_encode(tags, vocab))
    tag_vec = np.concatenate(tag_vecs)

    # Numeric features
    num_vec = numeric_features[row_idx].astype(np.float32)

    # Concatenate all features - Game profile vector
    profile = np.concatenate([desc_emb, com_emb, tag_vec, num_vec])
    profiles[game_id] = profile
  
  return profiles, encoder

def get_profile_dim(
  tag_vocabs: Dict[str, Dict[str, int]],
  cfg: EmbeddingConfig = EmbeddingConfig(),
  n_numeric: int = len(GAME_NUMERIC_FEATURES)
):
  tag_dim = sum(len(vocab) for vocab in tag_vocabs.values())
  return 2 * cfg.embedding_dim + tag_dim + n_numeric