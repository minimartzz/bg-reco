"""
Training Pipeline
========================================
Two-phase training:
1. Train the two-tower model with contrastive loss (InfoNCE)
2. Train the reranker on top-K candidates from the frozen towers

Training Data Construction
--------------------------
From records, every (user_id, game_id) pair that a user has actually played is a positive
pair. Negative pairs come from in-batch sampling (random selection of games not played yet)
and hard-negative mining (games that were not recommended)
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
import random
from typing import List, Tuple, Dict
from dataclasses import asdict
from torch.utils.data import DataLoader, Dataset
from config import ModelConfig, TrainingConfig

from model import TwoTowerModel, info_nce_loss, Reranker

# ========================================
# DATASETS
# ========================================
class TwoTowerDataset(Dataset):
  """
  Each sample is a (user_feature_vec, game_feature_vec) positive pair - a
  user has played the game before
  """
  def __init__(
    self,
    pairs: List[Tuple[int, int]],
    user_features: Dict[int, np.ndarray],
    game_features: Dict[int, np.ndarray]
  ):
    # Only get the pairs where there are game and user information
    self.pairs = [
      (u, g) for u, g in pairs
      if u in user_features and g in game_features
    ]
    self.user_features = user_features
    self.game_features = game_features

  def __len__(self) -> int:
    return len(self.pairs)
  
  def __getitem__(self, idx: int):
    user_id, game_id = self.pairs[idx]
    return (
      torch.tensor(self.user_features[user_id], dtype=torch.float32),
      torch.tensor(self.game_features[game_id], dtype=torch.float32)
    )


class RerankerDataset(Dataset):
  """
  Each sample is (user_tower_emb, game_tower_emb, label).
  Positives - Games that user has played before
  Negatives - Any random game user did not play
  """
  def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, float]]):
    self.samples = samples
  
  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, idx: int):
    user_emb, game_emb, label = self.samples[idx]
    return (
      torch.tensor(user_emb, dtype=torch.float32),
      torch.tensor(game_emb, dtype=torch.float32),
      torch.tensor(label, dtype=torch.float32),
    )


def build_reranker_data(
  two_tower: TwoTowerModel,
  pairs: List[Tuple[int, int]],
  user_features: Dict[int, np.ndarray],
  game_features: Dict[int, np.ndarray],
  all_game_ids: List[int],
  num_negatives: int,
  device: torch.device,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
  """
  Generate reranker training data by encoding users/games through
  the frozen towers and pairing with positive + negative labels.
  """
  two_tower.eval()
  samples = []

  game_embs: Dict[int, np.ndarray] = {}
  for gid, profile in game_features.items():
    t = torch.tensor(profile, dtype=torch.float32).unsqueeze(0).to(device)
    game_embs[gid] = two_tower.get_game_embedding(t).cpu().numpy().squeeze()

  user_embs: Dict[int, np.ndarray] = {}
  for uid, feats in user_features.items():
    t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    user_embs[uid] = two_tower.get_game_embedding(t).cpu().numpy().squeeze()
  
  user_positives: Dict[int, set] = {}
  for uid, gid in pairs:
    user_positives.setdefault(uid, set()).add(gid)
  
  for uid, gid in pairs:
    if uid not in user_embs or gid not in game_embs:
      continue
  
  u_emb = user_embs[uid]
  samples.append((u_emb, game_embs[gid], 1.0))

  negative_pools = [g for g in all_game_ids if g not in user_positives.get(uid, set()) and g in game_embs]
  neg_ids = random.sample(negative_pools, min(num_negatives, len(negative_pools)))
  for neg_gid in neg_ids:
    samples.append((u_emb, game_embs[neg_gid], 0.0))
  
  random.shuffle(samples)
  return samples


# ========================================
# TRAINING LOOPS
# ========================================
def extract_positive_pairs(records) -> List[Tuple[int, int]]:
  pairs = records[['profile_id', 'game_id']].drop_duplicates() # NOTE: This means games played mulitple times are not accounted for
  return list(zip(pairs['profile_id'].astype(int), pairs['game_id'].astype(int)))

def train_two_tower(
  model: TwoTowerModel,
  train_loader: DataLoader,
  val_loader: DataLoader,
  cfg: TrainingConfig,
  device: torch.device,
) -> TwoTowerModel:
  optimiser = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser,
    T_max=cfg.epochs
  )

  model.to(device)
  best_val_loss = float("inf")
  best_state = None

  for epoch in range(cfg.epochs):
    # Train
    model.train()
    train_losses = []
    for user_feats, game_feats in train_loader:
      # Skip if the batch has <2 users inside
      if user_feats.size(0) < 2:
        continue

      user_feats = user_feats.to(device)
      game_feats = game_feats.to(device)

      logits = model(user_feats, game_feats)
      loss = info_nce_loss(logits)

      optimiser.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimiser.step()
      train_losses.append(loss.item())
    
    scheduler.step()
    current_lr = scheduler.get_last_lr()

    # Validate
    model.eval()
    val_losses = []
    for user_feats, game_feats in val_loader:
      if user_feats.size(0) < 2:
        continue
      user_feats = user_feats.to(device)
      game_feats = game_feats.to(device)
      logits = model(user_feats, game_feats)
      val_losses.append(info_nce_loss(logits).item())
    
    avg_train = float(np.mean(train_losses)) if train_losses else float("inf")
    avg_val = float(np.mean(val_losses)) if val_losses else float("inf")

    # MLFlow Logging
    mlflow.log_metrics(
      {
        "two_tower/train_loss": avg_train,
        "two_tower/val_loss": avg_val,
        "two_tower/learning_rate": current_lr
      },
      step=epoch
    )

    if avg_val < best_val_loss:
      best_val_loss = avg_val
      best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if (epoch + 1) % 10 == 0:
      print(f"  [TOWER] Epoch {epoch+1:3d}/{cfg.epochs} | train_loss={avg_train:.4f},  val_loss={avg_val:.4f}")
  
  mlflow.log_metric("two_tower/best_val_loss", best_val_loss)

  if best_state:
    model.load_state_dict(best_state)
  return model


def train_reranker(
  reranker: Reranker,
  train_loader: DataLoader,
  cfg: TrainingConfig,
  device: torch.device
) -> Reranker:
  optimiser = torch.optim.AdamW(
    reranker.parameters(),
    lr=cfg.reranker_lr,
    weight_decay=cfg.weight_decay
  )
  criterion = nn.BCEWithLogitsLoss()
  reranker.to(device)

  for epoch in range(cfg.reranker_epochs):
    reranker.train()
    losses = []
    correct = 0
    total = 0

    for u_emb, g_emb, labels in train_loader:
      u_emb = u_emb.to(device)
      g_emb = g_emb.to(device)
      labels = labels.to(device)

      scores = reranker(u_emb, g_emb)
      loss = criterion(scores, labels)

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()
      losses.append(loss.item())

      # Track binary classification accuracy
      preds = (torch.sigmoid(scores) > 0.5).float()
      correct += (preds == labels).sum().item()
      total += labels.size(0)
    
    avg_loss = float(np.mean(losses))
    accuracy = correct / total if total > 0 else 0.0

    mlflow.log_metrics(
      {
        "reranker/train_loss": avg_loss,
        "reranker/train_accuracy": accuracy
      },
      step=epoch
    )

    if (epoch + 1) % 10 == 0:
      print(f"  [RERANKER] Epoch {epoch+1:3d}/{cfg.reranker_epochs} | train_loss={avg_loss:.4f},  val_loss={accuracy:.4f}")
  
  return reranker


# ========================================
# MLFLOW METRICS
# ========================================
def _log_dataclass_params(obj, prefix: str = "") -> None:
  """Log all fields of a dataclass as MLflow parameters"""
  for k, v in asdict(obj).items():
    param_name = f"{prefix}/{k}" if prefix else k
    mlflow.log_param(param_name, str(v)[:500])


def _log_data_summary(
  games: pd.DataFrame,
  records: pd.DataFrame,
  comments: pd.DataFrame,
  tag_vocabs: Dict[str, Dict[str, int]],
  profile_dim: int,
  user_dim: int
) -> None:
  mlflow.log_params({
    "data/n_games": len(games),
    "data/n_records": len(records),
    "data/n_comments": len(comments),
    "data/n_unique_users": records["profile_id"].nunique(),
    "data/n_unique_games_played": records["game_id"].nunique(),
    "data/profile_dim": profile_dim,
    "data/user_dim": user_dim,
    "data/tag_vocab_categories": len(tag_vocabs.get("categories", {})),
    "data/tag_vocab_mechanics": len(tag_vocabs.get("mechanics", {})),
    "data/tag_vocab_families": len(tag_vocabs.get("families", {})),
    "data/tag_vocab_artists": len(tag_vocabs.get("artists", {})),
    "data/tag_vocab_designers": len(tag_vocabs.get("designers", {})),
  })


# ========================================
# TRAINING ENTRYPOINT
# ========================================