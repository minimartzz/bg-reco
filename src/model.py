"""
Model Architecture
========================================

Two-Tower Retrieval Model
----------------------------------------
Two-Tower model maps users and games into a shared embedding space, where similarity
corresponds to preference. Each tower distills information for each profile type:

1. Game Tower: Game features -> [Game Tower MLP] -> Game Embedding (128d)
2. User Tower: User features -> [User Tower MLP] -> User Embedding (128d)
3. Score = Similarity(User Embedding, Game Embedding)

TRAINING OBJECTIVE
InfoNCE Loss - For each user-game pair, other games in the batch are treated as negative.
Then minimize the loss:

    L = -log( exp(sim(u, g+)/τ) / Σ_j exp(sim(u, g_j)/τ) )

Pushes the user embedding closer to the games that they are likely to play and away from
random games.

Cross-Encoder Reranker
----------------------------------------
Two-tower model can only capture interactions through dot product of independently computed
embeddings. The reranker is a cross-encoder that compares the user and game features simultaneously,
enabling richer feature interactions:

    [user_features ‖ game_features ‖ element-wise product] → MLP → score
  
The element-wise product (user_emb * game_emb) is a standard trick from factorisation machines and 
DeepFM — it explicitly encodes second-order feature interactions that the concatenation alone would
need multiple layers to learn.

Reranker is trained on the top-K candidates from the two-tower model with binary labels (played =1,
not played = 0) using BCE loss. At serving time it re-scores the ~50 candidates from retrieval and
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from config import ModelConfig

# ---- MLP --------------------
def _build_mlp(
  input_dim: int,
  hidden_dims: List[int],
  output_dim: int,
  dropout: float = 0.2
) -> nn.Sequential:
  """
  Constructs MLP with BatchNorm, ReLU, and Dropout betwen layers,
  specified by hidden_dims.

  Batch normalisation stabilises training when input features mix
  very different scales (384-d embeddings vs. binary tags vs. z-scored
  numerics).
  """
  layers: list = []
  prev = input_dim
  for h in hidden_dims:
    layers.extend([
      nn.Linear(prev, h),
      nn.BatchNorm1d(h),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout)
    ])
    prev = h
  layers.append(nn.Linear(prev, output_dim))
  return nn.Sequential(*layers)


# ---- TWO-TOWER --------------------
class GameTower(nn.Module):
  """Projects a game vector into the shared embedding space"""
  def __init__(
    self,
    input_dim: int,
    cfg: ModelConfig = ModelConfig()
  ):
    super().__init__()
    self.net = _build_mlp(
      input_dim,
      cfg.game_hidden_dims,
      cfg.tower_output_dim,
      cfg.dropout
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      emb = self.net(x)
      return F.normalize(emb, p=2, dim=-1)  # L2 normalise for cosine similarity


class UserTower(nn.Module):
  """Projects a user vector into the shared embedding space"""
  def __init__(
    self,
    input_dim: int,
    cfg: ModelConfig = ModelConfig()
  ):
    super().__init__()
    self.net = _build_mlp(
      input_dim,
      cfg.user_hidden_dims,
      cfg.tower_output_dim,
      cfg.dropout
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      emb = self.net(x)
      return F.normalize(emb, p=2, dim=-1)  # L2 normalise for cosine similarity


class TwoTowerModel(nn.Module):
  def __init__(
    self,
    game_input_dim: int,
    user_input_dim: int,
    cfg: ModelConfig = ModelConfig()
  ):
    super().__init__()
    self.game_tower = GameTower(game_input_dim, cfg)
    self.user_tower = UserTower(user_input_dim, cfg)
    self.temperature = cfg.temperature
  
  def forward(
    self,
    user_features: torch.Tensor,
    game_features: torch.Tensor
  ) -> torch.Tensor:
    """
    Computes similarity scores between projected users and games pairs in the batch
    """
    user_emb = self.user_tower(user_features) # (B, D)
    game_emb = self.game_tower(game_features) # (B, D)
    logits = torch.matmul(user_emb, game_emb.T) / self.temperature # (B, B)
    return logits
  
  def get_user_embedding(self, user_features: torch.Tensor) -> torch.Tensor:
    self.eval()
    with torch.no_grad():
      return self.user_tower(user_features)

  def get_game_embedding(self, game_features: torch.Tensor) -> torch.Tensor:
    self.eval()
    with torch.no_grad():
      return self.game_tower(game_features)


# ---- RERANKER --------------------
class Reranker(nn.Module):
  """
  Cross-encoder reranker that scores (user, game) pairs.

  Input: concatenation of [user_emb, game_emb, user_emb * game_emb]
  where user_emb and game_emb are the tower outputs (128-d each).
  """
  def __init__(
    self,
    tower_dim: int,
    cfg: ModelConfig = ModelConfig()
  ):
    super().__init__()
    input_dim = tower_dim * 3 # user_emb, game_emb, element-wise product
    self.net = _build_mlp(
      input_dim,
      cfg.reranker_hidden_dims,
      1,
      cfg.dropout
    )

    def forward(
      self,
      user_emb: torch.Tensor,
      game_emb: torch.Tensor,
    ) -> torch.Tensor:
      """
      Creates the element-wise interactions between user embeddings and games,
      then runs it through the MLP

      Args:
          user_emb (torch.Tensor): (batch, tower_output_dim)
          game_emb (torch.Tensor): (batch, tower_output_dim)

      Returns:
          torch.Tensor: (batch,) game-user relevance scores
      """
      interactions = user_emb * game_emb
      combined = torch.cat([user_emb, game_emb, interactions], dim=-1)
      return self.net(combined).squeeze(-1) # (B,) relevance scores


# ---- LOSS FUNCTION --------------------
def info_nce_loss(logits: torch.Tensor) -> torch.Tensor:
  """
  InfoNCE loss: For a batch of B (user, game) pairs, the positive for user_i is
  game_i (diagonal). All other games are negative
  """
  labels = torch.arange(logits.size(0), device=logits.device)
  loss_u2g = F.cross_entropy(logits, labels)
  loss_g2u = F.cross_entropy(logits.T, labels)
  return (loss_u2g + loss_g2u) / 2