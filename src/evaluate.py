"""
Evaluation Pipeline
===================
Scores the recommendaiton system using standard information-retrieval metrics
that measure retrieval quality and ranking quality.

Evaluation Strategy: Leave-One-Out
----------------------------------
For each user with >=2 sessions:
  1. Use one game that the user has played as the target
  2. Build the user profile with the remaining games
  3. Ask the model for the recommendations
  4. Check where the target game is in those recommendations

Metrics Used
------------
Refer to the notebook that describes each metric
  - Recall@K
  - NDCG@K
  - MRR
  - HitRate@K
  - MAP@K
  - Coverage@K

Two Version of Evaluation
-------------------------
1. Retrieval only: Top-k from the retrieval stage (two-tower embeddings).
   Informs us whether the fast approximate stage is finding the right candidates
2. Full pipeline (retrieval + reranker): Evaluate the final output. Tells us
   whether the reranker is improving the quality of scores
"""
import os
import torch
import json
import pickle
import numpy as np
import pandas as pd
import random
import mlflow
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from config import ModelConfig, DataConfig
from model import TwoTowerModel, Reranker
from preprocessing import load_data
from user_builder import build_user_profile
load_dotenv()

# ========================================
# METRIC FUNCTIONS
# ========================================
def recall_at_k(ranked_list: List[int], relevant: set, k: int) -> float:
  """Fraction of relevant items found in the top-k results"""
  if not relevant:
    return 0.0
  hits = len(set(ranked_list[:k]) & relevant)
  return hits / len(relevant)


def ndcg_at_k(ranked_list: List[int], relevant: set, k: int) -> float:
  """
  NDCG - Normalised Discounted Cumulative Gain

  Since comparing only a single relevant item (leave-one-out), this
  equals 1/log₂(rank+1) if the item is in top-k, else 0
  """
  dcg = 0.0
  for i, item in enumerate(ranked_list[:k]):
    if item in relevant:
      dcg += 1.0 / np.log2(i + 2)
  
  # Ideal DCG: all relevant items at the top
  idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
  if idcg == 0:
    return 0.0
  return dcg / idcg


def mrr(ranked_list: List[int], relevant: set, k: int) -> float:
  """Reciprocal rank of the first relevant itm in top-k"""
  for i, item in enumerate(ranked_list[:k]):
    if item in relevant:
      return 1.0 / (i + 1)
  return 0.0


def average_precision_at_k(ranked_list: List[int], relevant: set, k: int) -> float:
  """Precision at each relevant hit averaged"""
  if not relevant:
    return 0.0
  hits = 0
  sum_precision = 0.0
  for i, item in enumerate(ranked_list[:k]):
    if item in relevant:
      hits += 1
      sum_precision += hits / (i + 1)
  return sum_precision / min(len(relevant), k)


def compute_all_metrics(ranked_list: List[int], relevant: set, k: int) -> Dict[str, float]:
  """Compute all per-user metrics for a single recommendation list"""
  return {
    f"recall@{k}": recall_at_k(ranked_list, relevant, k),
    f"ndcg@{k}": ndcg_at_k(ranked_list, relevant, k),
    f"mrr@{k}": mrr(ranked_list, relevant, k),
    f"hit@{k}": 1.0 if set(ranked_list[:k]) & relevant else 0.0,
    f"map@{k}": average_precision_at_k(ranked_list, relevant, k),
  }


# ========================================
# EVALUATOR
# ========================================
class Evaluator:
  def __init__(self, artefact_dir: str = "models"):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.artefact_dir = artefact_dir

    # ---- Load Metadata -----------------
    with open(f"{artefact_dir}/meta.json") as f:
      meta = json.load(f)
    self.profile_dim = meta['profile_dim']
    self.user_dim = meta['user_dim']
    self.tower_output_dim = meta['tower_output_dim']

    model_cfg = ModelConfig(tower_output_dim=self.tower_output_dim)

    # ---- Load Two-tower Model ----------
    self.two_tower = TwoTowerModel(self.profile_dim, self.user_dim, model_cfg)
    self.two_tower.load_state_dict(
      torch.load(f"{artefact_dir}/two_tower.pt", map_location=self.device, weights_only=True)
    )
    self.two_tower.to(self.device)
    self.two_tower.eval()

    # ---- Load Reranker -----------------
    self.reranker = Reranker(self.tower_output_dim, model_cfg)
    self.reranker.load_state_dict(
      torch.load(f"{artefact_dir}/reranker.pt", map_location=self.device, weights_only=True)
    )
    self.reranker.to(self.device)
    self.reranker.eval()

    # ---- Load Pre-computed Embeddings --
    with open(f"{artefact_dir}/game_index.json", "rb") as f:
      raw = json.load(f)
    self.game_ids = [int(k) for k in raw.keys()]
    self.game_embeddings = np.array(
      [raw[str(gid)] for gid in self.game_ids], dtype=np.float32
    )

    # ---- Load Game Profiles -------------
    with open(f"{artefact_dir}/game_profiles.pkl", "rb") as f:
      self.game_profiles: Dict[int, np.ndarray] = pickle.load(f)
  
  def _get_user_embeddings(self, user_feature = np.ndarray) -> np.ndarray:
    t = torch.tensor(user_feature, dtype=torch.float32).unsqueeze(0).to(self.device)
    with torch.no_grad():
      emb = self.two_tower.get_user_embedding(t)
    return emb.cpu().numpy().squeeze(0)
  
  def _retrieval_rank(
    self, user_emb: np.ndarray, exclude_ids: set, k: int
  ) -> List[int]:
    """Ranks all games by two-tower dot product, excluding some IDs. Return top-k game IDs"""
    scores = self.game_embeddings @ user_emb
    ranked_indices = np.argsort(scores)[::-1]

    results = []
    for idx in ranked_indices:
      gid = self.game_ids[idx]
      if gid not in exclude_ids:
        results.append(gid)
      if len(results) >= k:
        break
    return results
  
  def _rerank(self, user_emb: np.ndarray, candidate_ids: List[int]) -> List[int]:
    """Re-score and re-sort candidates with the cross-encoder reranker"""
    if not candidate_ids:
      return []
    
    # Look up game embeddings for candidates
    id_to_idx = {gid: i for i, gid in enumerate(self.game_ids)}
    candidate_game_embs = np.array(
      [self.game_embeddings[id_to_idx[gid]] for gid in candidate_ids if gid in id_to_idx],
      dtype=np.float32
    )
    valid_ids = [gid for gid in candidate_ids if gid in id_to_idx]

    if len(valid_ids) == 0:
      return []

    user_emb_repeated = np.tile(user_emb, (len(valid_ids), 1))

    with torch.no_grad():
      u_t = torch.tensor(user_emb_repeated, dtype=torch.float32).to(self.device)
      g_t = torch.tensor(candidate_game_embs, dtype=torch.float32).to(self.device)
      rerank_score = self.reranker(u_t, g_t).cpu().numpy()
    
    ranked_indices = np.argsort(rerank_score)[::-1]
    return [valid_ids[i] for i in ranked_indices]
  
  def evaluate_user(
    self,
    user_records: pd.DataFrame,
    held_out_game_id: int,
    k_values: List[int],
    retrieval_k: int = 50,
  ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Evaluate a single user with one held-out game.

    Returns:
      (retrieval metrics, full_pipeline_metrics) 
      Each is {stage: {metric_name: value}}
    """
    # User features from "training" history
    remaining = user_records[user_records['game_id'] != held_out_game_id]
    user_feat = build_user_profile(remaining, self.game_profiles, self.profile_dim)
    user_emb = self._get_user_embeddings(user_feat)

    relevant = {held_out_game_id}
    played = set(remaining['game_id'].unique().tolist())

    # Retrieval Evaluation
    retrieval_list = self._retrieval_rank(user_emb, exclude_ids=played, k=max(retrieval_k, max(k_values)))

    retrieval_metrics: Dict[str, float] = {}
    for k in k_values:
      metrics = compute_all_metrics(retrieval_list, relevant, k)
      retrieval_metrics.update({f"retrieval/{name}": val for name, val in metrics.items()})

    # Pipeline Evaluation
    candidates = self._retrieval_rank(user_emb, exclude_ids=played, k=retrieval_k)
    reranked_list = self._rerank(user_emb, candidates)

    pipeline_metrics: Dict[str, float] = {}
    for k in k_values:
      metrics = compute_all_metrics(reranked_list, relevant, k)
      pipeline_metrics.update({f"pipeline/{name}": val for name, val in metrics.items()})
    
    return retrieval_metrics, pipeline_metrics
  
  def run_evaluation(
    self,
    records: pd.DataFrame,
    k_values: List[int] = [5, 10],
    retrieval_k: int = 50,
    seed: int = 43
  ) -> Dict[str, float]:
    """
    Run evaluation loop across all eligible users

    Eligible user: >=2 distinct games (1 for history, 1 for holdout)

    Returns: {metrics_name: value}
    """
    rng = random.Random(seed)

    # Group records by user
    user_groups = records.groupby("profile_id")

    # Collect per-user metric dicts
    all_retrieval: List[Dict[str, float]] = []
    all_pipeline: List[Dict[str, float]] = []
    all_recommended_ids: set = set()
    n_eligible = 0

    for uid, group in user_groups:
      # Check if user has min 2 unique games
      unique_games = group['game_id'].unique().tolist()
      if len(unique_games) < 2:
        continue

      n_eligible += 1
      # Randomly select a game as the hold out
      held_out = rng.choice(unique_games)

      retrieval_m, pipeline_m = self.evaluate_user(
        group,
        held_out,
        k_values,
        retrieval_k
      )
      all_retrieval.append(retrieval_m)
      all_pipeline.append(pipeline_m)

      # Track coverage (which games got recommended?)
      remaining = group[group['game_id'] != held_out]
      played = set(remaining['game_id'].unique().tolist())
      user_feat = build_user_profile(remaining, self.game_profiles, self.profile_dim)
      user_emb = self._get_user_embeddings(user_feat)
      top_recs = self._retrieval_rank(user_emb, exclude_ids=played, k=max(k_values))
      all_recommended_ids.update(top_recs[:max(k_values)])
    
    # ---- Macro-average all metrics ------------------
    results: Dict[str, float] = {}
    results['n_eligible_users'] = n_eligible

    if not all_retrieval:
      print("  [WARN] No eligible users for evaluation (need ≥ 2 games each)")
      return results
    
    # Average retrieval metrics
    all_keys = all_retrieval[0].keys()
    for key in all_keys:
      values = [m[key] for m in all_retrieval]
      results[key] = float(np.mean(values))

    # Average pipeline metrics
    all_keys = all_pipeline[0].keys()
    for key in all_keys:
      values = [m[key] for m in all_pipeline]
      results[key] = float(np.mean(values))

    # Coverage: fraction of catalogue recommended across all users
    total_games = len(self.game_ids)
    for k in k_values:
      results[f"coverage@{k}"] = len(all_recommended_ids) / total_games if total_games > 0 else 0.0

    return results


# ========================================
# ENTRYPOINT
# ========================================
def run_eval(
  data_cfg: DataConfig = DataConfig(),
  artefact_dir: str = "models",
  k_values: List[int] = [5, 10],
  retrieval_k: int = 50,
  experiment_name: str = "bg_recommender",
  log_to_mlflow: bool = True,
) -> Dict[str, float]:
  print("=" * 60)
  print("Board Game Recommendation — Evaluation")
  print("=" * 60)

  # Load data
  print("\n[1/3] Loading data...")
  _, records, _ = load_data(data_cfg)
  print(f"  {len(records)} records, {records['profile_id'].nunique()} users")

  # Build evaluator
  print("\n[2/3] Loading trained artifacts...")
  evaluator = Evaluator(artefact_dir)
  print(f"  {len(evaluator.game_ids)} games in index")

  # Run evaluation
  print(f"\n[3/3] Running evaluation (K = {k_values})...")
  results = evaluator.run_evaluation(records, k_values=k_values, retrieval_k=retrieval_k)

  # ---- Display Results -----------------------
  print(f"\n{'─' * 60}")
  print(f"  Eligible users: {results.get('n_eligible_users', 0)}")
  print(f"{'─' * 60}")

  for stage in ["retrieval", "pipeline"]:
    stage_label = "Two-Tower (retrieval only)" if stage == "retrieval" else "Full Pipeline (retrieval + reranker)"
    print(f"\n  {stage_label}")
    print(f"  {'·' * 50}")
    for k in k_values:
      print(f"    K = {k}:")
      for metric_short in ["recall", "ndcg", "mrr", "hit", "map"]:
        key = f"{stage}/{metric_short}@{k}"
        val = results.get(key, 0.0)
        print(f"      {metric_short.upper():>8}@{k}: {val:.4f}")
  
  # Coverage
  print(f"\n  Catalogue Coverage")
  print(f"  {'·' * 50}")
  for k in k_values:
    key = f"coverage@{k}"
    val = results.get(key, 0.0)
    print(f"    Coverage@{k}: {val:.4f}")
  
  # ---- Log to MLFlow ---------------------------
  if log_to_mlflow:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MLFLOW_USER")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MLFLOW_PASSWORD")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="evaluation"):
      mlflow.log_params({
        "eval/k_values": str(k_values),
        "eval/retrieval_k": retrieval_k,
        "eval/n_eligible_users": results.get("n_eligible_users", 0),
        "eval/artefact_dir": artefact_dir,
      })

      metric_results = {k: v for k, v in results.items() if k != "n_eligible_users"}
      mlflow_metrics = {
        k.replace("@", "_at_"): v for k, v in metric_results.items()
      }
      mlflow.log_metrics(mlflow_metrics)

      print(f"\n  Metrics logged to MLFlow experiment '{experiment_name}'")

  return results

if __name__ == "__main__":
  run_eval()