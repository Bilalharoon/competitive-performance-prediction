import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
# Config
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = BASE_DIR / "../../data/processed"
INPUT_FILE = PROCESSED_DATA_DIR / "pre_match_features.csv"

def train_lightgbm():
    print("Loading dataset...")
    if not INPUT_FILE.exists():
        print(f"Error: Dataset not found at {INPUT_FILE}")
        return
    df = pd.read_csv(INPUT_FILE)
    # Ensure temporal ordering before split
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'])
        df = df.sort_values(by='match_date').reset_index(drop=True)
    print(f"Dataset shape: {df.shape}")
    categorical_cols = ['division', 'stage']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    columns_to_drop = ['label', 'match_id', 'fighter_id', 'opponent_id', 'match_date']
    # Ensure we only drop columns that actually exist
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=columns_to_drop),
        df['label'],
        test_size=0.2,
        shuffle=False
    )
    print(f"Train set size: {x_train.shape[0]}")
    print(f"Test set size: {x_test.shape[0]}")
    lgb_train = lgb.Dataset(x_train, label=y_train, categorical_feature=categorical_cols)
    lgb_eval = lgb.Dataset(x_test, label=y_test, reference=lgb_train, categorical_feature=categorical_cols)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], callbacks=[lgb.early_stopping(stopping_rounds=10)])
    y_pred = model.predict(x_test, num_iteration=model.best_iteration)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_binary))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")
if __name__ == "__main__":
    train_lightgbm()

