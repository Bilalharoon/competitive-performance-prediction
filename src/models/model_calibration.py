"""
Model Calibration Testing

This script evaluates the calibration of predicted probabilities from both
Logistic Regression and LightGBM models. Calibration measures how well
predicted probabilities match observed frequencies.

Well-calibrated models: When a model predicts 70% probability, the actual
outcome should occur ~70% of the time.
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
import json
import matplotlib.pyplot as plt

# Config
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = BASE_DIR / "../../data/processed"
INPUT_FILE = PROCESSED_DATA_DIR / "pre_match_features.csv"
OUTPUT_DIR = BASE_DIR / "../../data/processed"
OUTPUT_FILE = OUTPUT_DIR / "model_calibration_results.json"


def calculate_ece(y_true, y_pred_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the average difference between predicted probabilities
    and observed frequencies across bins.
    
    Args:
        y_true: True binary labels
        y_pred_prob: Predicted probabilities
        n_bins: Number of bins for discretization
    
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_pred_prob > bin_lower) & (y_pred_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            # Calculate average predicted probability in this bin
            avg_confidence_in_bin = y_pred_prob[in_bin].mean()
            # Add weighted difference
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_mce(y_true, y_pred_prob, n_bins=10):
    """
    Calculate Maximum Calibration Error (MCE).
    
    MCE measures the maximum difference between predicted probabilities
    and observed frequencies across all bins.
    
    Args:
        y_true: True binary labels
        y_pred_prob: Predicted probabilities
        n_bins: Number of bins for discretization
    
    Returns:
        MCE value
    """
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
    """
    Comprehensive calibration analysis.
    
    Returns a dictionary with calibration metrics and curve data.
    """
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_prob, n_bins=n_bins, strategy='uniform'
    )
    
    # Calculate metrics
    ece = calculate_ece(y_true, y_pred_prob, n_bins=n_bins)
    mce = calculate_mce(y_true, y_pred_prob, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_pred_prob)
    logloss = log_loss(y_true, y_pred_prob)
    
    # Calculate bin statistics
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_info = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_pred_prob > bin_lower) & (y_pred_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_info.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'n_samples': int(in_bin.sum()),
                'mean_predicted_prob': float(y_pred_prob[in_bin].mean()),
                'mean_observed_freq': float(y_true[in_bin].mean()),
                'calibration_error': float(abs(y_pred_prob[in_bin].mean() - y_true[in_bin].mean()))
            })
    
    results = {
        'model_name': model_name,
        'metrics': {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier),
            'log_loss': float(logloss),
            'n_samples': len(y_true)
        },
        'calibration_curve': {
            'fraction_of_positives': [float(x) for x in fraction_of_positives],
            'mean_predicted_value': [float(x) for x in mean_predicted_value]
        },
        'bin_statistics': bin_info
    }
    
    return results


def train_logistic_regression_model(df):
    """Train Logistic Regression model and return predictions."""
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
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    return y_test, y_pred_prob, 'Logistic Regression'


def train_lightgbm_model(df):
    """Train LightGBM model and return predictions."""
    categorical_cols = ['division', 'stage']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    columns_to_drop = ['label', 'match_id', 'fighter_id', 'opponent_id', 'match_date']
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    
    X = df.drop(columns=columns_to_drop)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Process categorical columns for LightGBM
    cat_cols = [c for c in categorical_cols if c in X_train.columns]
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    for col in cat_cols:
        if col in X_train_processed.columns:
            X_train_processed[col] = X_train_processed[col].astype('category')
        if col in X_test_processed.columns:
            X_test_processed[col] = X_test_processed[col].astype('category')
    
    lgb_train = lgb.Dataset(X_train_processed, label=y_train, categorical_feature=cat_cols)
    lgb_eval = lgb.Dataset(X_test_processed, label=y_test, reference=lgb_train, categorical_feature=cat_cols)
    
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
        valid_sets=[lgb_train, lgb_eval], 
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    y_pred_prob = model.predict(X_test_processed, num_iteration=model.best_iteration)
    
    return y_test, y_pred_prob, 'LightGBM'


def print_calibration_report(results):
    """Print a formatted calibration report."""
    print("\n" + "=" * 80)
    print("CALIBRATION REPORT")
    print("=" * 80)
    
    for result in results:
        metrics = result['metrics']
        model_name = result['model_name']
        
        print(f"\n{model_name}:")
        print(f"  Expected Calibration Error (ECE): {metrics['ece']:.4f}")
        print(f"  Maximum Calibration Error (MCE):  {metrics['mce']:.4f}")
        print(f"  Brier Score:                      {metrics['brier_score']:.4f}")
        print(f"  Log Loss:                         {metrics['log_loss']:.4f}")
        print(f"  Test Samples:                     {metrics['n_samples']}")
        
        # Interpretation
        print(f"\n  Interpretation:")
        if metrics['ece'] < 0.05:
            ece_quality = "Excellent"
        elif metrics['ece'] < 0.10:
            ece_quality = "Good"
        elif metrics['ece'] < 0.20:
            ece_quality = "Fair"
        else:
            ece_quality = "Poor"
        
        print(f"    ECE Quality: {ece_quality}")
        print(f"    Lower is better for all metrics.")
        
        # Show worst calibrated bins
        bin_stats = sorted(result['bin_statistics'], 
                          key=lambda x: x['calibration_error'], 
                          reverse=True)
        
        if len(bin_stats) > 0:
            worst_bin = bin_stats[0]
            print(f"\n  Worst Calibrated Bin:")
            print(f"    Predicted: {worst_bin['mean_predicted_prob']:.3f}, "
                  f"Observed: {worst_bin['mean_observed_freq']:.3f}, "
                  f"Error: {worst_bin['calibration_error']:.3f}")
            print(f"    Range: [{worst_bin['bin_lower']:.2f}, {worst_bin['bin_upper']:.2f}], "
                  f"Samples: {worst_bin['n_samples']}")


def plot_calibration_curves(results, save_path=None):
    """
    Plot calibration curves for all models.
    
    Args:
        results: List of calibration analysis results
        save_path: Optional path to save the plot
    """
    try:
        plt.figure(figsize=(10, 8))
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Plot each model's calibration curve
        for result in results:
            curve = result['calibration_curve']
            plt.plot(
                curve['mean_predicted_value'],
                curve['fraction_of_positives'],
                marker='o',
                label=result['model_name'],
                linewidth=2
            )
        
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nCalibration plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    except Exception as e:
        print(f"Warning: Could not generate calibration plot: {e}")
        print("(This is non-critical - metrics are still calculated)")


def main():
    print("=" * 80)
    print("MODEL CALIBRATION TESTING")
    print("=" * 80)
    
    # Load data
    print("\nLoading dataset...")
    if not INPUT_FILE.exists():
        print(f"Error: Dataset not found at {INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    
    # Ensure temporal ordering before split
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'])
        df = df.sort_values(by='match_date').reset_index(drop=True)
    
    print(f"Dataset shape: {df.shape}")
    
    all_results = []
    
    # Analyze Logistic Regression
    print("\n" + "-" * 80)
    print("Analyzing Logistic Regression calibration...")
    print("-" * 80)
    try:
        y_test_lr, y_pred_prob_lr, model_name_lr = train_logistic_regression_model(df)
        lr_results = analyze_calibration(y_test_lr, y_pred_prob_lr, model_name_lr)
        all_results.append(lr_results)
        print("✓ Logistic Regression calibration analysis complete")
    except Exception as e:
        print(f"✗ Error analyzing Logistic Regression: {e}")
    
    # Analyze LightGBM
    print("\n" + "-" * 80)
    print("Analyzing LightGBM calibration...")
    print("-" * 80)
    try:
        # Create a fresh copy for LightGBM (to avoid dtype issues)
        df_lgbm = pd.read_csv(INPUT_FILE)
        if 'match_date' in df_lgbm.columns:
            df_lgbm['match_date'] = pd.to_datetime(df_lgbm['match_date'])
            df_lgbm = df_lgbm.sort_values(by='match_date').reset_index(drop=True)
        
        y_test_lgbm, y_pred_prob_lgbm, model_name_lgbm = train_lightgbm_model(df_lgbm)
        lgbm_results = analyze_calibration(y_test_lgbm, y_pred_prob_lgbm, model_name_lgbm)
        all_results.append(lgbm_results)
        print("✓ LightGBM calibration analysis complete")
    except Exception as e:
        print(f"✗ Error analyzing LightGBM: {e}")
    
    # Print report
    if all_results:
        print_calibration_report(all_results)
        
        # Generate plot
        plot_path = OUTPUT_DIR / "calibration_curves.png"
        plot_calibration_curves(all_results, save_path=plot_path)
        
        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to {OUTPUT_FILE}")
    else:
        print("\nNo results to save - all analyses failed.")


if __name__ == '__main__':
    main()
