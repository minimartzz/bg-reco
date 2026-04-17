"""
Inference Test
==============
Test script for the inference engine
"""
import pandas as pd
from inference import RecommendationEngine
from typing import List
from config import DataConfig
from preprocessing import load_data

def display_users(records: pd.DataFrame) -> List[int]:
  return records['profile_id'].unique().tolist()

def test_inference(
  profile_id: int,
  records: pd.DataFrame,
  top_k: int = 10,
  retrieval_top_k: int = 50,
  exclude_played: bool = True
):
  engine = RecommendationEngine("models")
  play_history = records[records['profile_id'] == profile_id]

  if play_history.empty:
    print(f"[ERROR] No profile with ID: {profile_id} found.")
    return []

  recommendations = engine.recommend(
    play_history,
    top_k,
    retrieval_top_k,
    exclude_played
  )

  return recommendations

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
    prog="Test Inference Engine",
    description="Testing the inferencce pipeline by passing a profile ID"
  )
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-u', '--user', type=int, help="User to show recommendations for")
  args = parser.parse_args()

  _, records, _ = load_data(DataConfig())

  if args.show:
    ids = display_users(records)
    print(ids)
  else:
    recommendations = test_inference(args.user, records)
    print(recommendations)
