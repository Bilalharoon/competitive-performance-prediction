"""
Compare model win probabilities with precalculated win_chance feature.

This script:
1. Loads tournament_histories.csv (with win_chance) and pre_match_features.csv
2. Trains logistic regression and LightGBM models
3. Gets predictions from both models
4. Compares all three: win_chance, logistic regression, lightgbm
5. Calculates reliability curves and metrics (Brier score, ECE, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Config
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "../../data/raw"
PROCESSED_DATA_DIR = BASE_DIR / "../../data/processed"
INPUT_FILE = PROCESSED_DATA_DIR / "pre_match_features.csv"
MERGED_MATCHES_FILE = PROCESSED_DATA_DIR / "merged_matches.csv"
OUTPUT_DIR = PROCESSED_DATA_DIR
OUTPUT_FILE = OUTPUT_DIR / "winchance_comparison_results.json"
PLOT_FILE = OUTPUT_DIR / "winchance_reliability_curves.png"


def calculate_ece(y_true, y_pred_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_prob > bin_lower) & (y_pred_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_mce(y_true, y_pred_prob, n_bins=10):
    """Calculate Maximum Calibration Error (MCE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_prob > bin_lower) & (y_pred_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_prob[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def analyze_calibration(y_true, y_pred_prob, model_name, n_bins=10):
    """Comprehensive calibration analysis."""
    # Remove any NaN or invalid predictions
    valid_mask = ~(np.isnan(y_pred_prob) | np.isinf(y_pred_prob))
    y_true_valid = y_true[valid_mask]
    y_pred_prob_valid = y_pred_prob[valid_mask]
    
    if len(y_true_valid) == 0:
        return None
    
    # Ensure probabilities are in [0, 1]
    y_pred_prob_valid = np.clip(y_pred_prob_valid, 0, 1)
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_valid, y_pred_prob_valid, n_bins=n_bins, strategy='uniform'
    )
    
    # Calculate metrics
    ece = calculate_ece(y_true_valid, y_pred_prob_valid, n_bins=n_bins)
    mce = calculate_mce(y_true_valid, y_pred_prob_valid, n_bins=n_bins)
    brier = brier_score_loss(y_true_valid, y_pred_prob_valid)
    logloss = log_loss(y_true_valid, y_pred_prob_valid)
    
    results = {
        'model_name': model_name,
        'metrics': {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier),
            'log_loss': float(logloss),
            'n_samples': len(y_true_valid)
        },
        'calibration_curve': {
            'fraction_of_positives': [float(x) for x in fraction_of_positives],
            'mean_predicted_value': [float(x) for x in mean_predicted_value]
        }
    }
    
    return results


def convert_winchance_to_prob(win_chance_str):
    """Convert win_chance string (e.g., '72%') to probability (0.72)."""
    if pd.isna(win_chance_str) or win_chance_str == '':
        return np.nan
    try:
        # Remove % and convert to float
        win_chance_str = str(win_chance_str).strip().replace('%', '')
        return float(win_chance_str) / 100.0
    except (ValueError, AttributeError):
        return np.nan


def train_and_predict_logistic_regression(df_features, df_eval):
    """Train logistic regression model and return predictions on eval set."""
    feature_cols = [
        'ratings_diff', 
        'experience_diff', 
        'fighter_first_match', 
        'opponent_first_match', 
        'days_since_last_fought_diff', 
        'fighter_days_since_last_fought', 
        'opponent_days_since_last_fought'
    ]
    
    # Use all available data for training
    X_train = df_features[feature_cols]
    y_train = df_features['label']
    
    # Get features for evaluation
    X_eval = df_eval[feature_cols]
    
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_eval)[:, 1]
    
    return y_pred_prob


def train_and_predict_lightgbm(df_features, df_eval):
    """Train LightGBM model and return predictions on eval set."""
    categorical_cols = ['division', 'stage']
    
    # Prepare training data
    columns_to_drop = ['label', 'match_id', 'fighter_id', 'opponent_id', 'match_date']
    columns_to_drop = [c for c in columns_to_drop if c in df_features.columns]
    
    X_train = df_features.drop(columns=columns_to_drop)
    y_train = df_features['label']
    
    # Prepare evaluation data (same columns)
    X_eval = df_eval.drop(columns=[c for c in columns_to_drop if c in df_eval.columns])
    
    # Process categorical columns
    cat_cols = [c for c in categorical_cols if c in X_train.columns]
    X_train_processed = X_train.copy()
    X_eval_processed = X_eval.copy()
    
    for col in cat_cols:
        if col in X_train_processed.columns:
            X_train_processed[col] = X_train_processed[col].astype('category')
        if col in X_eval_processed.columns:
            X_eval_processed[col] = X_eval_processed[col].astype('category')
    
    # Ensure all columns match
    missing_cols = set(X_train_processed.columns) - set(X_eval_processed.columns)
    if missing_cols:
        for col in missing_cols:
            X_eval_processed[col] = 0  # Fill missing columns with 0
    
    X_eval_processed = X_eval_processed[X_train_processed.columns]
    
    lgb_train = lgb.Dataset(X_train_processed, label=y_train, categorical_feature=cat_cols)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    model = lgb.train(
        params, 
        lgb_train, 
        valid_sets=[lgb_train], 
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    y_pred_prob = model.predict(X_eval_processed, num_iteration=model.best_iteration)
    
    return y_pred_prob


def main():
    print("=" * 80)
    print("COMPARING MODELS WITH WIN_CHANCE FEATURE")
    print("=" * 80)
    
    # Load data
    print("\nLoading datasets...")
    df_features = pd.read_csv(INPUT_FILE)
    df_merged = pd.read_csv(MERGED_MATCHES_FILE)
    
    # Convert dates
    if 'match_date' in df_features.columns:
        df_features['match_date'] = pd.to_datetime(df_features['match_date'])
        df_features = df_features.sort_values(by='match_date').reset_index(drop=True)
    
    print(f"Features dataset shape: {df_features.shape}")
    print(f"Merged matches dataset shape: {df_merged.shape}")
    
    # Merge to get win_chance
    print("\nMerging datasets...")
    df_eval = df_features.merge(
        df_merged[['match_id', 'win_chance']], 
        on='match_id', 
        how='inner'
    )
    
    print(f"Merged dataset shape: {df_eval.shape}")
    
    # Filter to rows with win_chance
    print("\nFiltering to rows with win_chance...")
    df_eval = df_eval[df_eval['win_chance'].notna() & (df_eval['win_chance'] != '')].copy()
    print(f"Rows with win_chance: {df_eval.shape[0]}")
    
    if len(df_eval) == 0:
        print("ERROR: No rows with win_chance found!")
        return
    
    # Convert win_chance to probability
    df_eval['win_chance_prob'] = df_eval['win_chance'].apply(convert_winchance_to_prob)
    
    # Remove rows where conversion failed
    df_eval = df_eval[df_eval['win_chance_prob'].notna()].copy()
    print(f"Valid win_chance values: {df_eval.shape[0]}")
    
    if len(df_eval) == 0:
        print("ERROR: No valid win_chance probabilities found!")
        return
    
    # Get true labels
    y_true = df_eval['label'].values
    
    # Get win_chance predictions
    winchance_probs = df_eval['win_chance_prob'].values
    
    print("\n" + "-" * 80)
    print("Training models...")
    print("-" * 80)
    
    # Train and get predictions from logistic regression
    print("Training Logistic Regression...")
    try:
        lr_probs = train_and_predict_logistic_regression(df_features, df_eval)
        print(f"✓ Logistic Regression: {len(lr_probs)} predictions")
    except Exception as e:
        print(f"✗ Error training Logistic Regression: {e}")
        lr_probs = None
    
    # Train and get predictions from LightGBM
    print("Training LightGBM...")
    try:
        lgbm_probs = train_and_predict_lightgbm(df_features, df_eval)
        print(f"✓ LightGBM: {len(lgbm_probs)} predictions")
    except Exception as e:
        print(f"✗ Error training LightGBM: {e}")
        lgbm_probs = None
    
    # Analyze calibration for each
    print("\n" + "-" * 80)
    print("Analyzing calibration...")
    print("-" * 80)
    
    all_results = []
    
    # Analyze win_chance
    print("Analyzing win_chance...")
    winchance_results = analyze_calibration(y_true, winchance_probs, 'win_chance')
    if winchance_results:
        all_results.append(winchance_results)
        print("✓ win_chance analysis complete")
    
    # Analyze logistic regression
    if lr_probs is not None:
        print("Analyzing Logistic Regression...")
        lr_results = analyze_calibration(y_true, lr_probs, 'Logistic Regression')
        if lr_results:
            all_results.append(lr_results)
            print("✓ Logistic Regression analysis complete")
    
    # Analyze LightGBM
    if lgbm_probs is not None:
        print("Analyzing LightGBM...")
        lgbm_results = analyze_calibration(y_true, lgbm_probs, 'LightGBM')
        if lgbm_results:
            all_results.append(lgbm_results)
            print("✓ LightGBM analysis complete")
    
    # Print results
    print("\n" + "=" * 80)
    print("CALIBRATION RESULTS")
    print("=" * 80)
    
    # Sort by ECE (lower is better)
    all_results_sorted = sorted(all_results, key=lambda x: x['metrics']['ece'])
    
    for i, result in enumerate(all_results_sorted):
        metrics = result['metrics']
        model_name = result['model_name']
        rank = i + 1
        
        print(f"\n{rank}. {model_name}:")
        print(f"   Expected Calibration Error (ECE):  {metrics['ece']:.6f}")
        print(f"   Maximum Calibration Error (MCE):   {metrics['mce']:.6f}")
        print(f"   Brier Score:                       {metrics['brier_score']:.6f}")
        print(f"   Log Loss:                          {metrics['log_loss']:.6f}")
        print(f"   Samples:                           {metrics['n_samples']}")
    
    # Determine best model
    if all_results_sorted:
        best_model = all_results_sorted[0]
        print(f"\n{'='*80}")
        print(f"BEST MODEL (Lowest ECE): {best_model['model_name']}")
        print(f"ECE: {best_model['metrics']['ece']:.6f}")
        print(f"{'='*80}")
    
    # Plot calibration curves
    print("\nGenerating reliability curves...")
    try:
        plt.figure(figsize=(10, 8))
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Plot each model's calibration curve
        for result in all_results_sorted:
            curve = result['calibration_curve']
            plt.plot(
                curve['mean_predicted_value'],
                curve['fraction_of_positives'],
                marker='o',
                label=f"{result['model_name']} (ECE: {result['metrics']['ece']:.4f})",
                linewidth=2
            )
        
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives (Observed)', fontsize=12)
        plt.title('Reliability Curves (Calibration Curves)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {PLOT_FILE}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating plot: {e}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results_sorted, f, indent=2)
    print(f"✓ Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
