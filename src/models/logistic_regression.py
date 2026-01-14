
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from pathlib import Path

# Config
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = BASE_DIR / "../../data/processed"
INPUT_FILE = PROCESSED_DATA_DIR / "pre_match_features.csv"

def train_logistic_regression():
    print("Loading dataset...")
    if not INPUT_FILE.exists():
        print(f"Error: Dataset not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # Features and Target
    # We drop metadata columns and categorical columns for the baseline
    # Identifiers: fighter_id, opponent_id, match_date
    # Categorical: division, stage (TODO: OneHotEncode these if needed)
    # Target: label
    
    feature_cols = [
        'ratings_diff', 
        'experience_diff', 
        'fighter_first_match', 
        'opponent_first_match', 
        'days_since_last_fought_diff', 
        'fighter_days_since_last_fought', 
        'opponent_days_since_last_fought'
    ]
    
    X = df[feature_cols]
    y = df['label']

    print(f"Features: {feature_cols}")
    print(f"Target: label")
    print(f"Dataset shape: {df.shape}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train model
    print("Training Logistic Regression model...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    
    # Create a pipeline with scaling and logistic regression
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nModel Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)

    # Feature Importance
    # Access the logistic regression step from the pipeline
    logistic_model = model.named_steps['logisticregression']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': logistic_model.coef_[0]
    }).sort_values(by='coefficient', ascending=False)
    
    print("\nFeature Importance (Scaled):")
    print(feature_importance)

if __name__ == '__main__':
    train_logistic_regression()
