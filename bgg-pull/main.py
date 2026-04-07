
import os
import requests
import pandas as pd
import numpy as np
import duckdb
import time
import argparse
import random
from xml.etree import ElementTree
from dotenv import load_dotenv

load_dotenv()

# ========================================
# Variables
# ========================================
TOKEN = os.environ['BGG_TOKEN']
DUCKDB_PATH = "../data/bgg_db.duckdb"
BGG_CSV_PATH = "../data/boardgames_ranks.csv"

# ========================================
# Helper Functions
# ========================================
def get_bgg_id(df: pd.DataFrame) -> int:
  """
  Get the highest ranking ID of the board game that hasn't been pulled yet

  Args:
    df (pd.DataFrame): boardgames_ranks df

  Returns:
    int: BGG ID
  """
  filtered_df = df[df['pulled'] != 1]
  return filtered_df['id'].to_numpy()[0]

def update_bgg_id(df:pd.DataFrame, id: int) -> pd.DataFrame:
  """
  Update the pulled column of the dataframe based on the BGG ID

  Args:
    df (pd.DataFrame): boardgames_ranks df
    id (int): BGG ID

  Returns:
    pd.DataFrame: Updated dataframe based on BGG ID
  """
  df.loc[df['id'] == id, 'pulled'] = 1
  return df

def get_suggested_players(tree):
  poll = tree.find(".//poll[@name='suggested_numplayers']")
  suggested = max(
    poll.findall("results"),
    key=lambda r: int(r.find("result[@value='Best']").get("numvotes"))
  )
  return suggested.get("numplayers")

def get_suggested_age(tree):
  poll = tree.find(".//poll[@name='suggested_playerage']")
  suggested = max(
    poll.findall(".//result"),
    key=lambda r: int(r.get("numvotes"))
  )
  return suggested.get("value")

def get_link_type_list(tree, tag):
  v = [l.get('value') for l in tree.findall(f".//link[@type='{tag}']")]
  i = [l.get('id') for l in tree.findall(f".//link[@type='{tag}']")]
  return i, v

# ========================================
# Load DuckDB store
# ========================================
def setup_duckdb(duckdb_path: str):
  con = duckdb.connect(duckdb_path)

  # Create schema
  con.execute(f"CREATE SCHEMA IF NOT EXISTS bgg;")

  # Create table 1: games
  print("Creating games table...")
  create_games_table = f"""
CREATE TABLE IF NOT EXISTS bgg.games (
  id INTEGER PRIMARY KEY,
  name VARCHAR,
  description VARCHAR,
  year_published VARCHAR,
  min_players INTEGER,
  max_players INTEGER,
  suggested_players VARCHAR,
  playing_time INTEGER,
  min_playing_time INTEGER,
  max_playing_time INTEGER,
  min_age INTEGER,
  suggested_age VARCHAR,
  category_ids INTEGER[],
  categories VARCHAR[],
  mechanic_ids INTEGER[],
  mechanics VARCHAR[],
  family_ids INTEGER[],
  families VARCHAR[],
  designer_ids INTEGER[],
  designers VARCHAR[],
  artist_ids INTEGER[],
  artists VARCHAR[],
  num_ratings INTEGER,
  rating DOUBLE,
  bayes_rating DOUBLE,
  complexity DOUBLE
);
  """
  con.execute(create_games_table)
  print("Games table successfully created.")

  # Create table 2: comments
  print("Creating comments table...")
  create_comments_table = f"""
CREATE TABLE IF NOT EXISTS bgg.comments (
  bgg_id INTEGER,
  rating DOUBLE,
  comment TEXT
);
  """
  con.execute(create_comments_table)
  print("Comments table successfully created.")

  return con

# ========================================
# Retrieve Game Details
# ========================================
def retrieve_game_info(bgg_id: int):
  url = f"https://boardgamegeek.com/xmlapi2/thing?id={bgg_id}&type=boardgame&stats=1&comments=1&page=1"

  print(f"[GAME] Attempting to retrieve game with id: {bgg_id}")
  response = requests.get(
    url,
    headers={"Authorization": f"Bearer {TOKEN}"}
  )

  if response.status_code == 200:
    tree = ElementTree.fromstring(response.content)
  else:
    print(f"[GAME] Error: Failed to retrieve game info")
    return None
  
  # Set bgg_info
  bgg_info = {}
  bgg_info['id'] = bgg_id
  bgg_info['name'] = tree.find(".//name[@type='primary']").get('value')
  bgg_info['description'] = tree.find(".//description").text.strip('\n')
  bgg_info['year_published'] = tree.find('.//yearpublished').get('value')
  ## Players
  bgg_info['min_players'] = int(tree.find(".//minplayers").get('value'))
  bgg_info['max_players'] = int(tree.find(".//maxplayers").get('value'))
  bgg_info['suggested_players'] = get_suggested_players(tree)
  ## Playing time
  bgg_info['playing_time'] = int(tree.find(".//playingtime").get('value'))
  bgg_info['min_playing_time'] = int(tree.find(".//minplaytime").get('value'))
  bgg_info['max_playing_time'] = int(tree.find(".//maxplaytime").get('value'))
  ## Age
  bgg_info['min_age'] = int(tree.find(".//minage").get('value'))
  bgg_info['suggested_age'] = get_suggested_age(tree)
  ## Tags
  bgg_info['category_ids'], bgg_info['categories'] = get_link_type_list(tree, 'boardgamecategory')
  bgg_info['mechanic_ids'], bgg_info['mechanics'] = get_link_type_list(tree, 'boardgamemechanic')
  bgg_info['family_ids'], bgg_info['families'] = get_link_type_list(tree, 'boardgamefamily')
  bgg_info['designer_ids'], bgg_info['designers'] = get_link_type_list(tree, 'boardgamedesigner')
  bgg_info['artist_ids'], bgg_info['artists'] = get_link_type_list(tree, 'boardgameartist')
  ## Ratings
  bgg_info['num_ratings'] = int(tree.find(".//usersrated").get('value'))
  bgg_info['rating'] = float(tree.find(".//average").get('value'))
  bgg_info['bayes_rating'] = float(tree.find(".//bayesaverage").get('value'))
  bgg_info['complexity'] = float(tree.find(".//averageweight").get('value'))
  
  print("[GAME] Successfully retrieved game info")
  return bgg_info

# ========================================
# Retrieve Comments
# ========================================
def retrieve_game_comments(
  bgg_id: int,
  page_range: int = 550,
  min_words: int = 15,
  max_pages: int = 20,
  min_comments: int = 30
):
  interval = 10
  comments = {
    "bgg_id": [],
    "rating": [],
    "comment": []
  }
  page_numbers = [i for i in range(1, page_range+1)]
  searched_pages = 1
  it = 0

  print(f"[COMMENT] Attempting to retrieve game comments with id: {bgg_id}.")
  while searched_pages < max_pages:
    time.sleep(interval)
    it += 1
    page = random.choice(page_numbers)
    url = f"https://boardgamegeek.com/xmlapi2/thing?id={bgg_id}&type=boardgame&ratingcomments=1&page={page}"
    print(f"    Iteration: {it} | Page: {page} | Number of Comments: {len(comments['comment'])} | Number of pages searched: {searched_pages}")

    response = requests.get(
      url,
      headers={"Authorization": f"Bearer {TOKEN}"}
    )

    if response.status_code == 200:
      tree = ElementTree.fromstring(response.content)
    else:
      print(f"[COMMENT] Error: Failed to retrieve game comments")
      continue

    # Check if there are any comments on the page
    all_comments = tree.findall('.//comment')
    if not all_comments:
      print("[COMMENT] No comments found on this page.")
      continue

    for comment in all_comments:
      text = comment.get('value').strip()
      if len(text.split(' ')) < min_words:
        continue
    
      rating = comment.get('rating')
      text = comment.get('value')
      
      if rating == 'N/A':
        rating = np.nan
      comments['bgg_id'].append(bgg_id)
      comments['rating'].append(float(rating) if isinstance(rating, str) else rating)
      comments['comment'].append(text)
    
    # Exit condition
    searched_pages += 1

    if len(comments['comment']) > min_comments:
      break
    

  return comments

# ========================================
# Insert to DB and Update DF
# ========================================
def insert_and_update(bgg_id: int, con, game_info: dict, game_comments: dict, df: pd.DataFrame):
  # Convert both dicts to pandas dfs
  game_info = {k: [v] for k, v in game_info.items()}
  game_info_df = pd.DataFrame.from_dict(game_info)
  game_comments = pd.DataFrame.from_dict(game_comments)

  # Try to insert
  try:
    print(f"[INSERT] Inserting game info of {bgg_id} into DB")
    con.execute("INSERT INTO bgg.games SELECT * FROM game_info_df")

    print(f"[INSERT] Inserting game comments of {bgg_id} into DB")
    con.execute("INSERT INTO bgg.comments SELECT * FROM game_comments")
  except:
    raise Exception("[INSERT] ERROR: Failed to insert into database")
  
  # Update dataframe
  df = update_bgg_id(df, bgg_id)
  return df

# ========================================
# Main Loop
# ========================================
def main(
  duckdb_path: str,
  bgg_csv_path: str,
  num_games_to_ingest: int = 1,
  page_range: int = 550,
  min_words: int = 15,
  max_pages: int = 20,
  min_comments: int = 30,
  specific_bgg_id: int = None
):
  con = setup_duckdb(duckdb_path)
  df = pd.read_csv(bgg_csv_path)

  if specific_bgg_id:
    print(f"[BGG] Pulling single game id: {specific_bgg_id}")
    game_info = retrieve_game_info(specific_bgg_id)
    game_comments = retrieve_game_comments(
      specific_bgg_id,
      page_range,
      min_words,
      max_pages,
      min_comments
    )
    df = insert_and_update(specific_bgg_id, con, game_info, game_comments, df)
  else:
    for _ in range(num_games_to_ingest):
      bgg_id = get_bgg_id(df)
      game_info = retrieve_game_info(bgg_id)
      game_comments = retrieve_game_comments(
        bgg_id,
        page_range,
        min_words,
        max_pages,
        min_comments
      )
      df = insert_and_update(bgg_id, con, game_info, game_comments, df)
  
  df.to_csv(bgg_csv_path, index=False)
  con.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog="BGG Data Puller",
    description="Pull BGG data based on the boardgames_rank.csv starting from the most popular entries"
  )
  parser.add_argument('-n', '--num', type=int, default=1, help="Number of games to ingest starting from the most popular. (Default 1)")
  parser.add_argument('-p', '--pages', type=int, default=20, help="Maximum number of comment pages to search through (Default 20)")
  parser.add_argument('-r', '--range', type=int, default=300, help="Range of comment pages to randomly search across. (Default 300)")
  parser.add_argument('-c', '--comments', type=int, default=30, help="Minimum number comments to collect (Default 30)")
  parser.add_argument('-w', '--words', type=int, default=15, help="Minimum number of words in a comment to be included (Default 15)")
  parser.add_argument('-i', '--id', type=int, help="Pull data for a specific BGG ID")
  args = parser.parse_args()

  main(
    DUCKDB_PATH,
    BGG_CSV_PATH,
    num_games_to_ingest=args.num,
    page_range=args.range,
    min_words=args.words,
    max_pages=args.pages,
    min_comments=args.comments,
    specific_bgg_id=args.id
  )
