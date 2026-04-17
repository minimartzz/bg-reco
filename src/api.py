"""
API Recommendation Server
=========================
Entrypoint for the recommendation engine built with FastAPI

Endpoints:
  - POST /recommend - Returns top-k game IDs given user's play history
  - GET  /health    - Liveness check
"""
import pandas as pd
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from inference import RecommendationEngine

# ========================================
# PYDANTIC SCHEMA
# ========================================
class PlayRecord(BaseModel):
  """A single session record for a user"""
  game_id: int
  date_played: str
  game_weight: float  = 3.0
  game_length: int    = 60
  num_players: int    = 4
  is_winner: bool     = False
  score: float        = 1.0
  is_first_play: bool = False

class RecommendRequest(BaseModel):
  """
  Request body for the /recommend endpoint 
  """
  play_history: List[PlayRecord] = Field(
    ..., min_length=1, description="User's play history records"
  )
  top_k: int = Field(
    default=5, ge=1, le=20, description="Number of recommendations"
  )
  exclude_played: bool = Field(
    default=True, description="Exclude games the user has already played"
  )

class GameRecommendation(BaseModel):
  game_id: int
  score: float

class RecommendResponse(BaseModel):
  recommendations: List[GameRecommendation]
  num_returned: int


# ========================================
# APP
# ========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
  """
  Lifespan object for preloading the model when the app starts
  """
  # Runs once on startup
  app.state.engine = RecommendationEngine("models")
  print(f"[ENGINE] Loaded {app.state.engine.n_games} games indexed")
  yield
  # Below runs when app stops
  print("[ENGINE] Shutting down...")

app = FastAPI(
  title="Board Game Recommendation API",
  description="Two-tower retrieval + cross-encoder rereanking recommendation system for board games",
  version="1.0.0"
)

@app.get("/health")
def health():
  return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request):
  engine = request.app.state.engine

  # Convert payload to a DataFrame
  records_dicts = [r.model_dumps() for r in req.play_history]
  history = pd.DataFrame(records_dicts)

  # Ensure that the payload as all the columns
  for col, default in [
    ("profile_id", 0),
    ("session_id", "api_request"),
    ("victory_points", 0),
    ("position", 0),
    ("is_vp", False),
    ("is_tie", False)
  ]:
    if col not in history.columns:
      history[col] = default
  
  try:
    results = engine.recommend(
      play_history=history,
      top_k=req.top_k,
      exclude_played=req.exclude_played
    )
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")
  
  recs = [GameRecommendation(game_id=r['game_id'], score=r['score']) for r in results]
  return RecommendResponse(recommendations=recs, num_returned=len(recs))

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)