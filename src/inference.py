"""
Inference Pipeline
==================
Provides the recommendation engine for the API to call the model.

Two-stage recommendation:
  1. Retrieval: Gets the top-k nearest game embeddings via dot product
                against the pre-computed game index
  2. Reranking: Feeds the top-k (user_emb, game_emb) pairs through the
                reranker and return the final top-X

API Workflow                
------------
  1. Build the user feature vector on-the-fly from their history.
  2. Run it through the user tower to get the user embedding.
  3. Dot-product against all pre-indexed game embeddings.
  4. Take top-K candidates.
  5. Score each candidate with the reranker.
  6. Return top-X game IDs sorted by reranker score.
"""
import torch
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
from config import ModelConfig
from model import TwoTowerModel, Reranker
from user_builder import build_user_profile

class RecommendationEngine:
  """
  Loads trained artefacts and serve recommendations

  Components:
    engine: RecommendationEnginer("models/") 
    game_ids = engine.recommend(play_history, top_k=10)
  """
  def __init__(self, artefact_dir: str = "models"):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.artefact_dir = Path(artefact_dir)
    self._load_artefacts()
  
  def _load_artefacts(self):
    """Load all trained models and precomputed data"""
    # Metadata
    with open(self.artefact_dir / "meta.json") as f:
      meta = json.load(f)
    self.profile_dim = meta["profile_dim"]
    self.user_dim = meta["user_dim"]
    self.tower_output_dim = meta["tower_output_dim"]

    model_cfg = ModelConfig(tower_output_dim=self.tower_output_dim)

    # Two-tower model
    self.two_tower = TwoTowerModel(self.profile_dim, self.user_dim, model_cfg)
    self.two_tower.load_state_dict(
      torch.load(self.artefact_dir / "two_tower.pt", map_location=self.device, weights_only=True)
    )
    self.two_tower.to(self.device)
    self.two_tower.eval()

    # Reranker
    self.reranker = Reranker(self.tower_output_dim, model_cfg)
    self.reranker.load_state_dict(
      torch.load(self.artefact_dir / "reranker.pt", map_location=self.device, weights_only=True)
    )
    self.reranker.to(self.device)
    self.reranker.eval()

    # Pre-computed game embeddings (game tower outputs)
    with open(self.artefact_dir / "game_index.json") as f:
      raw = json.load(f)
    self.game_ids = [int(k) for k in raw.keys()]
    self.game_embeddings = np.array(
      [raw[str(gid)] for gid in self.game_ids], dtype=np.float32
    ) # (n_games, tower_dim)

    # Game profiles (raw feature vectors — needed to build user features)
    with open(self.artefact_dir / "game_profiles.pkl", "rb") as f:
        self.game_profiles: Dict[int, np.ndarray] = pickle.load(f)

    # Tag vocabularies and scaler (for encoding new games if needed)
    with open(self.artefact_dir / "tag_vocabs.pkl", "rb") as f:
        self.tag_vocabs = pickle.load(f)
    with open(self.artefact_dir / "scaler.pkl", "rb") as f:
        self.scaler = pickle.load(f)

    # Text encoder + embedding config (for encoding new games at serving time)
    with open(self.artefact_dir / "text_encoder.pkl", "rb") as f:
        self.text_encoder = pickle.load(f)
    with open(self.artefact_dir / "emb_cfg.pkl", "rb") as f:
        self.emb_cfg = pickle.load(f)
    
  def _build_user_feature_from_history(self, play_history: pd.DataFrame):
    return build_user_profile(
      play_history, self.game_profiles, self.profile_dim
    )

  def recommend(
    self,
    play_history: pd.DataFrame,
    top_k: int = 10,
    retrieval_top_k: int = 50,
    exclude_played: bool = True
  ) -> List[Dict]:
    """
    Generate top-k game recommendations for a user

    Args:
      play_history (pd.DataFrame): Dataframe of the user's play records.
      top_k (int, optional): Number of final recommendations. Defaults to 10.
      retrieval_top_k (int, optional): Candidates from the two-tower stage. Defaults to 50.
      exclude_played (bool, optional): Whether to exclude games already played. Defaults to True.

    Returns:
      List[Dict]: [{"game_id": int, "score": float}, ...]
                  sorted by descending reranker score
    """ 
    # ---- Build user features ------------------------------------
    user_feat = self._build_user_feature_from_history(play_history)
    user_tensor = torch.tensor(user_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

    # ---- Two-tower retrieval ------------------------------------
    with torch.no_grad():
      user_emb = self.two_tower.get_user_embedding(user_tensor) # (1, tower_dim)
    user_emb_np = user_emb.cpu().numpy().squeeze(0)

    # Scores against all games
    scores = self.game_embeddings @ user_emb_np # (n_games,)

    # Optional: Remove games user has played
    played_ids = set()
    if exclude_played and "game_id" in play_history.columns:
      played_ids = set(play_history['game_id'].unique().tolist())
    
    # Get top-k
    candidate_indices = np.argsort(scores)[::-1]
    candidates = []
    for idx in candidate_indices:
      gid = self.game_ids[idx]
      if gid not in played_ids:
        candidates.append((gid, idx))
      if len(candidates) >= retrieval_top_k:
        break
    
    if not candidates:
      return []

    # ---- Reranker ------------------------------------------------
    candidate_game_embs = np.array(
      [self.game_embeddings[idx] for _, idx in candidates], dtype=np.float32
    )
    user_emb_repeated = np.tile(user_emb_np, (len(candidates), 1))

    with torch.no_grad():
      u_t = torch.tensor(user_emb_repeated, dtype=torch.float32).to(self.device)
      g_t = torch.tensor(candidate_game_embs, dtype=torch.float32).to(self.device)
      rerank_scores = self.reranker(u_t, g_t).cpu().numpy()
    
    # Sort descending rank
    ranked_indices = np.argsort(rerank_scores)[::-1][:top_k]

    results = []
    for ri in ranked_indices:
      gid = candidates[ri][0]
      results.append({
         "game_id": gid,
         "score": float(rerank_scores[ri])
      })
    
    return results

