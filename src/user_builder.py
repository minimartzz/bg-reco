"""
User Profile Builder
========================================
Builds the user's input for the user tower by aggregating a user's play
history into a fixed-size vector. 2 components:

1. Weighted Game Profile Centroid: A weighted average of games that user has played.
Consists of 3 interaction signals:

  w_i = α · win_signal_i  +  β · score_signal_i  +  γ · recency_i

- **Win signal**: Games the user won receive higher weight. Winning a game suggests
  the user enjoyed playing the game (or at least winning it) so should have higher influence
- **Score signal**: Normalised score captures how well they performed regardless if they win
  or not. High score with a loss might still imply engagement
- **Recency**: More recent plays are weighted higher via exponential decay

Creates a game-profile centroid that represents the type of games the user is most engaged
with. NOTE: Adjusting the weights will shift the importance of each signal

2. Bahavioural Stats: Statistics that capture how the user plays
- Total games played (proxy for experience level)
- Win rate
- Average complexity
- Average game length
- Player count preference
- Diversity of games played

Measures the difference between a casual player (few games, low complexity) vs a power gamer 
(many games, high complexity, high win rate)
"""
import pandas as pd
import numpy as np
from typing import Dict

# ========================================
# PROFILE CENTROID
# ========================================
def compute_interaction_weights(
  user_records: pd.DataFrame,
  alpha: float = 0.3,
  beta: float = 0.3,
  gamma: float = 0.4,
  recency_halflife_days: float = 60.0
) -> np.ndarray:
  """
  Compute per-row interaction weights for a single user's play records

  Args:
    user_records (pd.DataFrame): A single players play history records
    alpha (float, optional): Weight for win signal. Defaults to 0.3.
    beta (float, optional): Weight for score signal. Defaults to 0.3.
    gamma (float, optional): Weight for recency signal. Defaults to 0.4.
    recency_halflife_days (float, optional): Halflife for recency decay. Defaults to 60.0.

  Returns:
    np.ndarray: An array of interaction weights for each play record
  """
  n = len(user_records)
  if n == 0:
    return np.array([], dtype=np.float32)
  
  # Win signal: 1 for win, 0.3 if loss
  win_signal = np.where(user_records['is_winner'], 1.0, 0.3)

  # Score signal: Min-max normalised score
  scores = user_records['victory_points'].fillna(0).values.astype(np.float64)
  score_min, score_max = scores.min(), scores.max()
  if score_max > score_min:
    score_signal = (scores - score_min) / (score_max - score_min)
  else:
    score_signal = np.ones(n) * 0.5 # All scores given the same weight
  
  # Recency signal: Exponential decay based on days since play
  dates = pd.to_datetime(user_records['date_played'], errors='coerce')
  max_date = dates.max()
  days_since_last_play = (max_date - dates).dt.total_seconds() / (3600.0 * 24)
  days_since_last_play = days_since_last_play.fillna(days_since_last_play.max() + 1).values
  recency_signal = np.exp(-np.log(2) * days_since_last_play / recency_halflife_days)

  # Normalise signal to sum to 1 with predefined weights
  raw_weights = alpha * win_signal + beta * score_signal + gamma * recency_signal
  total = raw_weights.sum()
  if total > 0:
    return (raw_weights / total).astype(np.float32)
  return np.ones(n, dtype=np.float32) / n


# ========================================
# BUILD USER PROFILE
# ========================================
def build_user_profile(
  user_records: pd.DataFrame,
  game_profiles: Dict[int, np.ndarray],
  profile_dim: int
) -> np.ndarray:
  """
  Build's a single user's feature vector from their play history and behavioural stats

  Args:
    user_records (pd.DataFrame): Single user's play history
    game_profiles (Dict[int, np.ndarray]): Encoded game profile
    profile_dim (int): Number of games played

  Returns:
    np.ndarray: (profile_dim + n_behavioural_stats)
  """
  n_behavioural = 6 # Represents the 6 summary statistics of a user

  if user_records.empty:
    return np.zeros(profile_dim + n_behavioural, dtype=np.float32)
  
  # Weighted Game Profile Centroid
  weights = compute_interaction_weights(user_records)
  centroid = np.zeros(profile_dim, dtype=np.float32)
  valid_weight_sum = 0.0

  for idx, (_, row) in enumerate(user_records.iterrows()):
    game_id = int(row['game_id'])
    if game_id in game_profiles:
      centroid += weights[idx] * game_profiles[game_id]
      valid_weight_sum += weights[idx]
  
  if valid_weight_sum > 0:
    centroid /= valid_weight_sum
  
  # Behavioural Stats
  total_plays = len(user_records)
  win_rate = user_records['is_winner'].mean()
  avg_complexity = user_records['game_weight'].fillna(0).mean()
  avg_length = user_records['game_length'].fillna(0).mean()
  avg_players = user_records['num_players'].fillna(0).mean()
  distinct_games = user_records['game_id'].nunique()

  behaviouoral_stats = np.array([
    total_plays,
    win_rate,
    avg_complexity,
    avg_length,
    avg_players,
    distinct_games
  ], dtype=np.float32)

  return np.concatenate([centroid, behaviouoral_stats])

def build_all_user_profiles(
  records: pd.DataFrame,
  game_profiles: Dict[int, np.ndarray],
  profile_dim: int
) -> Dict[int, np.ndarray]:
  """Build feature vectors for every user in the records"""
  user_profiles: Dict[int, np.ndarray] = {}
  for user_id, group in records.groupby("profile_id"):
    user_profiles[int(user_id)] = build_user_profile(group, game_profiles, profile_dim) 

  return user_profiles

def get_user_feature_dim(profile_dim: int) -> int:
  """Total user feature dimensionality."""
  return profile_dim + 6  # 6 behavioural stats