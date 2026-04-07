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
"""