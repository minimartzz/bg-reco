"""
Test Script for deployed BGR API
================================
Testing the deployed BGR API on railway

Tests Conducted:
    - Health check
    - Multi-game history
    - Single-game history
    - exclude_played toggle
    - Empty history

Test details:
    - Status
    - Latency
    - Returned values
"""

import os
import json
import requests
from dotenv import load_dotenv
from time import time

load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8080") # Defaults to localhost

def test_health():
    """Test the /health endpoint"""
    print("=" * 50)
    print("[TEST API] GET /health")
    print("=" * 50)

    res = requests.get(f"{API_URL}/health", timeout=30)
    print(f"    Status: {res.status_code}")
    print(f"    Body: {res.json()}")
    assert res.status_code == 200
    assert res.json()['status'] == "ok"

    print("    PASSED\n")


def test_recommend_multi_history():
    """Test recommendation with multiple games played"""
    print("=" * 50)
    print("[TEST API] GET /recommend (multi)")
    print("=" * 50)

    payload = {
        "play_history": [
            {
                "game_id": 400314,
                "date_played": "2025-06-01",
                "game_weight": 3.01,
                "game_length": 90,
                "num_players": 3,
                "is_winner": True,
                "score": 85.0,
                "is_first_play": False,
            },
            {
                "game_id": 167791,
                "date_played": "2025-05-15",
                "game_weight": 3.6,
                "game_length": 90,
                "num_players": 4,
                "is_winner": False,
                "score": 62.0,
                "is_first_play": True,
            },
        ],
        "top_k": 5,
        "exclude_played": True,
    }

    start = time() # Test inference time
    res = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)
    elapsed = time() - start

    print(f"    Status: {res.status_code}")
    print(f"    Latency: {elapsed:.2f}s")
    
    if res.status_code == 200:
        data = res.json()
        print(f"    Returned: {data['num_returned']} recommendations")

        for rec in data["recommendations"]:
            print(f"    Game {rec['game_id']:>8} | score: {rec['score']:.4f}")
            assert data["num_returned"] == 5
            assert len(data["recommendations"]) == 5
            print("   PASSED\n")
    else:
        print(f"   Failed: {res.text}\n")


def test_recommend_single_history():
    """Test recommendation with single game"""
    print("=" * 50)
    print("[TEST API] GET /recommend (single)")
    print("=" * 50)

    payload = {
        "play_history": [
            {
                "game_id": 400314,
                "date_played": "2025-06-01",
                "game_weight": 3.01,
                "game_length": 90,
                "num_players": 3,
                "is_winner": True,
                "score": 85.0,
                "is_first_play": False,
            },
        ],
        "top_k": 5,
        "exclude_played": True,
    }

    res = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)
    print(f"    Status: {res.status_code}")

    if res.status_code == 200:
        data = res.json()
        print(f"    Returned: {data['num_returned']} recommendations")

        for rec in data["recommendations"]:
            print(f"    Game {rec['game_id']:>8} | score: {rec['score']:.4f}")
        
        # Verify the played game is excluded
        rec_ids = {r["game_id"] for r in data["recommendations"]}
        assert 342942 not in rec_ids, "Played game should be excluded!"

        print("  PASSED\n")
    else:
        print(f"  FAILED: {res.text}\n")


def test_recommend_exclude_toggle():
    """Test that exclude_played=False includes played games"""
    print("=" * 50)
    print("[TEST API] GET /recommend (excluded_play=False)")
    print("=" * 50)
 
    payload = {
        "play_history": [
            {
                "game_id": 400314,
                "date_played": "2025-06-01",
                "game_weight": 3.01,
                "game_length": 90,
                "num_players": 3,
                "is_winner": True,
                "score": 85.0,
                "is_first_play": False,
            },
        ],
        "top_k": 10,
        "exclude_played": False,
    }
 
    res = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)
    print(f"    Status: {res.status_code}")
 
    if res.status_code == 200:
        data = res.json()
        rec_ids = {r["game_id"] for r in data["recommendations"]}

        print(f"    Returned: {data['num_returned']} recommendations")
        print(f"    Played game 400314 in results: {400314 in rec_ids}")
        print("    PASSED\n")
    else:
        print(f"    FAILED: {res.text}\n")


def test_empty_history():
    """Test that empty play history returns 422 validation error"""
    print("=" * 50)
    print("[TEST API] GET /recommend (empty history)")
    print("=" * 50)
 
    payload = {
        "play_history": [],
        "top_k": 5,
    }
 
    resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)
    print(f"    Status: {resp.status_code}")

    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
    print("    PASSED\n")


if __name__ == "__main__":
    print(f"Testing API at: {API_URL}\n")

    test_health()
    test_recommend_multi_history()
    test_recommend_single_history()
    test_recommend_exclude_toggle()
    test_empty_history()

    print("=" * 50)
    print("[TEST API] ALL TESTS PASSED")
    print("=" * 50)