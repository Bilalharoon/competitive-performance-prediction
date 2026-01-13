import pandas as pd
from src.fighter_state import FighterState
from datetime import datetime, timedelta
from pathlib import Path


# Config
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "../data/raw"
PROCESSED_DATA_DIR = BASE_DIR / "../data/processed"

MATCHES_FILE = RAW_DATA_DIR / "tournament_histories.csv"
RATINGS_HISTORY_FILE = RAW_DATA_DIR / "fighter_ratings_history.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "pre_match_features.csv"
MERGED_MATCHES_FILE = PROCESSED_DATA_DIR / "merged_matches.csv"


def convert_date(date_str):
    return datetime.strptime(date_str.strip(), "%B %Y").date()


def build_dataset():
    fighter_states: dict[int, FighterState] = {}
    rows = []
    print("loading data...")
    matches = pd.read_csv(MATCHES_FILE)    
    ratings = pd.read_csv(RATINGS_HISTORY_FILE)


    print('Preprocessing data...')
    matches['tournament_date'] = matches['tournament_date'].apply(convert_date)
    matches = matches.sort_values(by='tournament_date').reset_index(drop=True)

    ratings['tournament_date'] = ratings['date'].apply(convert_date)
    ratings = ratings.sort_values(by='tournament_date').reset_index(drop=True)

    print('Merging data...')
    ratings['date'] = ratings['date'].astype('datetime64[ns]')
    matches['tournament_date'] = matches['tournament_date'].astype('datetime64[ns]')
    ratings['date'] = ratings['date'].astype('datetime64[ns]')
    # Drop original opponent_rating to avoid duplicates after merge
    if 'opponent_rating' in matches.columns:
        matches = matches.drop(columns=['opponent_rating'])
    
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce').astype('float64')

    # Ensure IDs are integers for merging
    matches = matches.dropna(subset=['fighter_id', 'opponent_id'])
    matches['fighter_id'] = matches['fighter_id'].astype('int64')
    matches['opponent_id'] = matches['opponent_id'].astype('int64')
    ratings['fighter_id'] = ratings['fighter_id'].astype('int64')

    print(matches.columns)
    matches = pd.merge_asof(
        matches, 
        ratings[['fighter_id', 'rating', 'date']], 
        left_on='tournament_date', 
        right_on='date', 
        by='fighter_id',
        direction='backward',
        allow_exact_matches=False  # Strict inequality: ensure we don't peek at future ratings
    ).rename(columns={'rating': 'fighter_rating', 'date': 'fighter_rating_date'})

    # 3b. Get Opponent's (Fighter B) Rating
    # We must rename columns momentarily to merge on opponent_id
    matches = pd.merge_asof(
        matches,
        ratings[['fighter_id', 'rating', 'date']].rename(columns={'fighter_id': 'opponent_id'}),
        left_on='tournament_date',
        right_on='date',
        by='opponent_id',
        direction='backward',
        allow_exact_matches=False
    ).rename(columns={'rating': 'opponent_rating', 'date': 'opponent_rating_date'}) 
    
    matches['fighter_rating'] = matches['fighter_rating'].fillna(1000.0)
    matches['opponent_rating'] = matches['opponent_rating'].fillna(1000.0) 
    print(matches.head())
    print('Building dataset...')
    fighter_states: dict[int, FighterState] = {}
    for _, match in matches.iterrows():
        f_id = int(match['fighter_id'])
        o_id = int(match['opponent_id'])
        date = match['tournament_date']

        # Initialize if new
        if f_id not in fighter_states: fighter_states[f_id] = FighterState(f_id, match['fighter_rating'], date)
        if o_id not in fighter_states: fighter_states[o_id] = FighterState(o_id, match['opponent_rating'], date)

        a = fighter_states[f_id]
        b = fighter_states[o_id]
        
        # Calculate Time Stats (Days since last fight)
        days_since_a = a.days_since_last_fight(date)
        days_since_b = b.days_since_last_fight(date)
        
        val_days_a = days_since_a if days_since_a is not None else 365 # Default for debut
        val_days_b = days_since_b if days_since_b is not None else 365

        # We already have ratings from the vectorized merge step!
        # Just use match['fighter_rating'] directly.
        
        rows.append({
            'fighter_id': f_id,
            'opponent_id': o_id,
            'match_date': date,
            'ratings_diff': match['fighter_rating'] - match['opponent_rating'],
            'experience_diff': a.matches_fought - b.matches_fought,
            'days_since_last_fought_diff': val_days_a - val_days_b,
            'fighter_days_since_last_fought': val_days_a,
            'opponent_days_since_last_fought': val_days_b,
            'division': match['division'],
            'stage': match['stage'],
            'label': 1 if match['outcome'] == 'WIN' else 0
        })

        # Update History for next time they appear
        a.record_match(match['outcome'] == 'WIN', date)
        b.record_match(match['outcome'] != 'WIN', date)
    
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    matches.to_csv(MERGED_MATCHES_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}")
    print(f"Merged matches saved to {MERGED_MATCHES_FILE}")
    print(f'Number of rows: {len(df)}')
    print(f'Number of rows in merged matches: {len(matches)}')

if __name__ == '__main__':
    build_dataset()
