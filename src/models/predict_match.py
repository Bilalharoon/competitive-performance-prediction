"""
Predict win probability between two fighters using Logistic Regression.

Uses only the latest data (most recent ratings and match history) to compute
features. Includes fighter names in the output.

Usage:
    python -m src.models.predict_match <fighter_id_1> <fighter_id_2>

Example:
    python -m src.models.predict_match 711 1831
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path
import sys

# Config
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

MATCHES_FILE = RAW_DATA_DIR / "tournament_histories.csv"
RATINGS_HISTORY_FILE = RAW_DATA_DIR / "fighter_ratings_history.csv"
RANKINGS_FILE = RAW_DATA_DIR / "rankings_1.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "pre_match_features.csv"

FEATURE_COLS = [
    "ratings_diff",
    "experience_diff",
    "fighter_first_match",
    "opponent_first_match",
    "days_since_last_fought_diff",
    "fighter_days_since_last_fought",
    "opponent_days_since_last_fought",
]


def _load_id_to_name() -> dict:
    """Build fighter id -> name from rankings and tournament_histories."""
    id_to_name = {}

    # rankings_1.csv: id, name
    if RANKINGS_FILE.exists():
        try:
            r = pd.read_csv(RANKINGS_FILE)
            for _, row in r.iterrows():
                try:
                    i = int(float(row["id"]))
                    if pd.notna(row.get("name")):
                        id_to_name[i] = str(row["name"]).strip()
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass

    # tournament_histories: opponent_id -> opponent (name)
    if MATCHES_FILE.exists():
        try:
            m = pd.read_csv(MATCHES_FILE, usecols=["opponent_id", "opponent"])
            m = m.dropna(subset=["opponent_id", "opponent"])
            m["opponent_id"] = pd.to_numeric(m["opponent_id"], errors="coerce")
            m = m.dropna(subset=["opponent_id"])
            for oid, name in m[["opponent_id", "opponent"]].drop_duplicates("opponent_id").values:
                try:
                    i = int(oid)
                    if i not in id_to_name and pd.notna(name):
                        id_to_name[i] = str(name).strip()
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass

    return id_to_name


def _get_latest_rating(fighter_id: int, as_of: date, ratings: pd.DataFrame) -> float:
    """Latest rating for fighter at or before as_of. ratings: date (date), fighter_id, rating."""
    sub = ratings[(ratings["fighter_id"] == fighter_id) & (ratings["date"] <= as_of)]
    if len(sub) == 0:
        return 1000.0
    row = sub.loc[sub["date"].idxmax()]
    r = row.get("rating")
    if pd.isna(r):
        return 1000.0
    try:
        return float(r)
    except (ValueError, TypeError):
        return 1000.0


def _get_latest_history(fighter_id: int, as_of: date, matches: pd.DataFrame):
    """
    For fighter_id, using matches with tournament_date <= as_of:
    (matches_fought, wins, last_match_date).
    """
    # rows where fighter is fighter_id or opponent_id
    m = matches[
        ((matches["fighter_id"] == fighter_id) | (matches["opponent_id"] == fighter_id))
        & (matches["tournament_date"] <= pd.Timestamp(as_of))
    ]
    if len(m) == 0:
        return 0, 0, None

    matches_fought = len(m)
    wins = int(
        ((m["fighter_id"] == fighter_id) & (m["outcome"] == "WIN")).sum()
        + ((m["opponent_id"] == fighter_id) & (m["outcome"] == "LOSS")).sum()
    )
    last = m["tournament_date"].max()
    last_date = last.date() if hasattr(last, "date") else pd.Timestamp(last).date()
    return matches_fought, wins, last_date


def _build_latest_features(fid1: int, fid2: int, as_of: date, ratings: pd.DataFrame, matches: pd.DataFrame) -> dict:
    """Build feature dict for fid1 vs fid2 using only latest data as of as_of."""
    r1 = _get_latest_rating(fid1, as_of, ratings)
    r2 = _get_latest_rating(fid2, as_of, ratings)

    n1, w1, last1 = _get_latest_history(fid1, as_of, matches)
    n2, w2, last2 = _get_latest_history(fid2, as_of, matches)

    d1 = (as_of - last1).days if last1 else 365
    d2 = (as_of - last2).days if last2 else 365

    return {
        "ratings_diff": r1 - r2,
        "experience_diff": n1 - n2,
        "fighter_first_match": 1 if n1 == 0 else 0,
        "opponent_first_match": 1 if n2 == 0 else 0,
        "days_since_last_fought_diff": d1 - d2,
        "fighter_days_since_last_fought": d1,
        "opponent_days_since_last_fought": d2,
    }, {"rating": r1, "matches": n1, "wins": w1}, {"rating": r2, "matches": n2, "wins": w2}


def _train_lr_model() -> "Pipeline":
    """Train logistic regression on pre_match_features (all data for model)."""
    df = pd.read_csv(FEATURES_FILE)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df["label"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    return model


def predict_win_probability(fighter_id_1: int, fighter_id_2: int):
    """
    Predict win probability for fighter_1 vs fighter_2 using only latest data.
    Returns dict with win_probability, names, features, and stats.
    """
    fighter_id_1 = int(fighter_id_1)
    fighter_id_2 = int(fighter_id_2)

    if not RATINGS_HISTORY_FILE.exists():
        raise FileNotFoundError(f"Ratings file not found: {RATINGS_HISTORY_FILE}")
    if not MATCHES_FILE.exists():
        raise FileNotFoundError(f"Matches file not found: {MATCHES_FILE}")
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

    # Load and prepare ratings (date as actual date for comparison)
    ratings = pd.read_csv(RATINGS_HISTORY_FILE)
    ratings["date"] = pd.to_datetime(ratings["date"], format="%B %Y", errors="coerce").dt.date
    ratings = ratings.dropna(subset=["date"])
    ratings["fighter_id"] = pd.to_numeric(ratings["fighter_id"], errors="coerce").astype("Int64")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["fighter_id", "rating"])

    # Load matches
    matches = pd.read_csv(MATCHES_FILE)
    matches = matches.dropna(subset=["fighter_id", "opponent_id", "tournament_date"])
    matches["fighter_id"] = pd.to_numeric(matches["fighter_id"], errors="coerce").astype("Int64")
    matches["opponent_id"] = pd.to_numeric(matches["opponent_id"], errors="coerce").astype("Int64")
    matches = matches.dropna(subset=["fighter_id", "opponent_id"])
    matches["tournament_date"] = pd.to_datetime(matches["tournament_date"].astype(str).str.strip(), format="%B %Y", errors="coerce")
    matches = matches.dropna(subset=["tournament_date"])

    # as_of = latest date in match data
    as_of = matches["tournament_date"].max().date()

    # Build features from latest data only
    features, s1, s2 = _build_latest_features(fighter_id_1, fighter_id_2, as_of, ratings, matches)

    # Train model and predict
    model = _train_lr_model()
    X = pd.DataFrame([features])[FEATURE_COLS]
    win_prob = float(model.predict_proba(X)[0, 1])

    # Names
    id_to_name = _load_id_to_name()
    name1 = id_to_name.get(fighter_id_1, f"Fighter {fighter_id_1}")
    name2 = id_to_name.get(fighter_id_2, f"Fighter {fighter_id_2}")

    return {
        "fighter_1_id": fighter_id_1,
        "fighter_2_id": fighter_id_2,
        "fighter_1_name": name1,
        "fighter_2_name": name2,
        "as_of_date": str(as_of),
        "win_probability": win_prob,
        "features": features,
        "fighter_1_stats": {"rating": s1["rating"], "matches": s1["matches"], "wins": s1["wins"]},
        "fighter_2_stats": {"rating": s2["rating"], "matches": s2["matches"], "wins": s2["wins"]},
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.models.predict_match <fighter_id_1> <fighter_id_2>")
        print("Example: python -m src.models.predict_match 711 1831")
        sys.exit(1)

    fid1 = int(sys.argv[1])
    fid2 = int(sys.argv[2])

    try:
        r = predict_win_probability(fid1, fid2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("WIN PROBABILITY (logistic regression, latest data only)")
    print("=" * 70)
    print(f"\n{r['fighter_1_name']} (ID {r['fighter_1_id']})  vs  {r['fighter_2_name']} (ID {r['fighter_2_id']})")
    print(f"Data as of: {r['as_of_date']}")
    print(f"\nWin probability for {r['fighter_1_name']}: {r['win_probability']:.2%}")
    print(f"Win probability for {r['fighter_2_name']}: {1 - r['win_probability']:.2%}")
    print("\n" + "-" * 70)
    print(f"{r['fighter_1_name']}:  rating {r['fighter_1_stats']['rating']:.1f},  {r['fighter_1_stats']['matches']} matches,  {r['fighter_1_stats']['wins']} wins")
    print(f"{r['fighter_2_name']}:  rating {r['fighter_2_stats']['rating']:.1f},  {r['fighter_2_stats']['matches']} matches,  {r['fighter_2_stats']['wins']} wins")
    print("-" * 70)
    print("Features: ratings_diff={:.2f}, experience_diff={}, fighter_first_match={}, opponent_first_match={}, days_since_last_fought_diff={}, fighter_days_since_last_fought={}, opponent_days_since_last_fought={}".format(
        r["features"]["ratings_diff"],
        r["features"]["experience_diff"],
        r["features"]["fighter_first_match"],
        r["features"]["opponent_first_match"],
        r["features"]["days_since_last_fought_diff"],
        r["features"]["fighter_days_since_last_fought"],
        r["features"]["opponent_days_since_last_fought"],
    ))
    print("=" * 70)


if __name__ == "__main__":
    main()
