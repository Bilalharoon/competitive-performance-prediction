"""
Feature Ablation Study

This script systematically removes features from models to assess their importance.
For each feature removed, we train the model and measure performance degradation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
from pathlib import Path
from itertools import combinations
import json

# Config
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = BASE_DIR / "../../data/processed"
INPUT_FILE = PROCESSED_DATA_DIR / "pre_match_features.csv"
OUTPUT_DIR = BASE_DIR / "../../data/processed"
OUTPUT_FILE = OUTPUT_DIR / "feature_ablation_results.json"


def train_logistic_regression_with_features(X_train, X_test, y_train, y_test, feature_cols):
    """Train logistic regression and return metrics."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    return {
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'n_features': len(feature_cols)
    }


def train_lightgbm_with_features(X_train, X_test, y_train, y_test, feature_cols, categorical_cols):
    """Train LightGBM and return metrics."""
    # Filter categorical columns to only those in feature_cols
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    
    # Convert categorical columns to category dtype (required by LightGBM)
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
    
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
    
    acc = accuracy_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return {
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'n_features': len(feature_cols)
    }


def run_single_feature_ablation(df, model_type='both'):
    """
    Remove one feature at a time and measure performance.
    
    Args:
        df: DataFrame with features
        model_type: 'logistic', 'lightgbm', or 'both'
    """
    print("=" * 80)
    print("SINGLE FEATURE ABLATION")
    print("=" * 80)
    
    # Prepare data
    columns_to_drop = ['label', 'match_id', 'fighter_id', 'opponent_id', 'match_date']
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    
    # Logistic Regression features
    lr_feature_cols = [
        'ratings_diff', 
        'experience_diff', 
        'fighter_first_match', 
        'opponent_first_match', 
        'days_since_last_fought_diff', 
        'fighter_days_since_last_fought', 
        'opponent_days_since_last_fought'
    ]
    
    # LightGBM features (all except metadata)
    lgbm_feature_cols = [c for c in df.columns if c not in columns_to_drop]
    categorical_cols = ['division', 'stage']
    categorical_cols = [c for c in categorical_cols if c in lgbm_feature_cols]
    
    # Convert categorical columns to category dtype for LightGBM
    df_processed = df.copy()
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')
    
    # Split data
    X_lr = df[lr_feature_cols]
    X_lgbm = df_processed.drop(columns=columns_to_drop)
    y = df['label']
    
    X_lr_train, X_lr_test, y_train, y_test = train_test_split(
        X_lr, y, test_size=0.2, shuffle=False, random_state=42
    )
    X_lgbm_train, X_lgbm_test, _, _ = train_test_split(
        X_lgbm, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    results = {}
    
    # Baseline (all features)
    print("\nTraining baseline models with all features...")
    if model_type in ['logistic', 'both']:
        baseline_lr = train_logistic_regression_with_features(
            X_lr_train, X_lr_test, y_train, y_test, lr_feature_cols
        )
        results['logistic_regression'] = {
            'baseline': baseline_lr,
            'ablations': {}
        }
        print(f"  Logistic Regression Baseline - Accuracy: {baseline_lr['accuracy']:.4f}, ROC-AUC: {baseline_lr['roc_auc']:.4f}")
    
    if model_type in ['lightgbm', 'both']:
        baseline_lgbm = train_lightgbm_with_features(
            X_lgbm_train, X_lgbm_test, y_train, y_test, lgbm_feature_cols, categorical_cols
        )
        results['lightgbm'] = {
            'baseline': baseline_lgbm,
            'ablations': {}
        }
        print(f"  LightGBM Baseline - Accuracy: {baseline_lgbm['accuracy']:.4f}, ROC-AUC: {baseline_lgbm['roc_auc']:.4f}")
    
    # Ablate each feature
    print("\nAblating features one at a time...")
    
    if model_type in ['logistic', 'both']:
        print("\nLogistic Regression:")
        for feature in lr_feature_cols:
            print(f"  Removing: {feature}")
            remaining_features = [f for f in lr_feature_cols if f != feature]
            X_train_subset = X_lr_train[remaining_features]
            X_test_subset = X_lr_test[remaining_features]
            
            metrics = train_logistic_regression_with_features(
                X_train_subset, X_test_subset, y_train, y_test, remaining_features
            )
            
            # Calculate performance drop
            acc_drop = baseline_lr['accuracy'] - metrics['accuracy']
            roc_drop = baseline_lr['roc_auc'] - metrics['roc_auc']
            
            results['logistic_regression']['ablations'][feature] = {
                'metrics': metrics,
                'accuracy_drop': float(acc_drop),
                'roc_auc_drop': float(roc_drop)
            }
            
            print(f"    Accuracy: {metrics['accuracy']:.4f} (drop: {acc_drop:+.4f})")
            print(f"    ROC-AUC: {metrics['roc_auc']:.4f} (drop: {roc_drop:+.4f})")
    
    if model_type in ['lightgbm', 'both']:
        print("\nLightGBM:")
        for feature in lgbm_feature_cols:
            print(f"  Removing: {feature}")
            remaining_features = [f for f in lgbm_feature_cols if f != feature]
            X_train_subset = X_lgbm_train[remaining_features]
            X_test_subset = X_lgbm_test[remaining_features]
            
            metrics = train_lightgbm_with_features(
                X_train_subset, X_test_subset, y_train, y_test, remaining_features, categorical_cols
            )
            
            # Calculate performance drop
            acc_drop = baseline_lgbm['accuracy'] - metrics['accuracy']
            roc_drop = baseline_lgbm['roc_auc'] - metrics['roc_auc']
            
            results['lightgbm']['ablations'][feature] = {
                'metrics': metrics,
                'accuracy_drop': float(acc_drop),
                'roc_auc_drop': float(roc_drop)
            }
            
            print(f"    Accuracy: {metrics['accuracy']:.4f} (drop: {acc_drop:+.4f})")
            print(f"    ROC-AUC: {metrics['roc_auc']:.4f} (drop: {roc_drop:+.4f})")
    
    return results


def run_sequential_feature_ablation(df, model_type='both'):
    """
    Sequentially remove features in order of least importance.
    This helps identify redundant features.
    """
    print("\n" + "=" * 80)
    print("SEQUENTIAL FEATURE ABLATION")
    print("=" * 80)
    
    columns_to_drop = ['label', 'match_id', 'fighter_id', 'opponent_id', 'match_date']
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    
    lr_feature_cols = [
        'ratings_diff', 
        'experience_diff', 
        'fighter_first_match', 
        'opponent_first_match', 
        'days_since_last_fought_diff', 
        'fighter_days_since_last_fought', 
        'opponent_days_since_last_fought'
    ]
    
    lgbm_feature_cols = [c for c in df.columns if c not in columns_to_drop]
    categorical_cols = ['division', 'stage']
    categorical_cols = [c for c in categorical_cols if c in lgbm_feature_cols]
    
    # Convert categorical columns to category dtype for LightGBM
    df_processed = df.copy()
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')
    
    X_lr = df[lr_feature_cols]
    X_lgbm = df_processed.drop(columns=columns_to_drop)
    y = df['label']
    
    X_lr_train, X_lr_test, y_train, y_test = train_test_split(
        X_lr, y, test_size=0.2, shuffle=False, random_state=42
    )
    X_lgbm_train, X_lgbm_test, _, _ = train_test_split(
        X_lgbm, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    results = {}
    
    if model_type in ['logistic', 'both']:
        print("\nLogistic Regression Sequential Ablation:")
        remaining = lr_feature_cols.copy()
        sequential_results = []
        
        while len(remaining) > 1:
            best_removal = None
            best_metrics = None
            min_drop = float('inf')
            
            # Try removing each remaining feature
            for feature in remaining:
                test_features = [f for f in remaining if f != feature]
                X_train_subset = X_lr_train[test_features]
                X_test_subset = X_lr_test[test_features]
                
                metrics = train_logistic_regression_with_features(
                    X_train_subset, X_test_subset, y_train, y_test, test_features
                )
                
                # Use ROC-AUC as primary metric
                if metrics['roc_auc'] < min_drop:
                    min_drop = metrics['roc_auc']
                    best_removal = feature
                    best_metrics = metrics
            
            if best_removal:
                remaining.remove(best_removal)
                sequential_results.append({
                    'removed': best_removal,
                    'remaining_features': remaining.copy(),
                    'metrics': best_metrics
                })
                print(f"  Removed {best_removal}, ROC-AUC: {best_metrics['roc_auc']:.4f}, Features: {len(remaining)}")
        
        results['logistic_regression_sequential'] = sequential_results
    
    if model_type in ['lightgbm', 'both']:
        print("\nLightGBM Sequential Ablation:")
        remaining = lgbm_feature_cols.copy()
        sequential_results = []
        
        while len(remaining) > 1:
            best_removal = None
            best_metrics = None
            min_drop = float('inf')
            
            for feature in remaining:
                test_features = [f for f in remaining if f != feature]
                X_train_subset = X_lgbm_train[test_features]
                X_test_subset = X_lgbm_test[test_features]
                
                metrics = train_lightgbm_with_features(
                    X_train_subset, X_test_subset, y_train, y_test, test_features, categorical_cols
                )
                
                if metrics['roc_auc'] < min_drop:
                    min_drop = metrics['roc_auc']
                    best_removal = feature
                    best_metrics = metrics
            
            if best_removal:
                remaining.remove(best_removal)
                sequential_results.append({
                    'removed': best_removal,
                    'remaining_features': remaining.copy(),
                    'metrics': best_metrics
                })
                print(f"  Removed {best_removal}, ROC-AUC: {best_metrics['roc_auc']:.4f}, Features: {len(remaining)}")
        
        results['lightgbm_sequential'] = sequential_results
    
    return results


def print_summary(results):
    """Print a summary of feature importance based on ablation results."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 80)
    
    if 'logistic_regression' in results:
        print("\nLogistic Regression - Feature Importance (by ROC-AUC drop):")
        lr_results = results['logistic_regression']
        ablations = lr_results['ablations']
        
        # Sort by ROC-AUC drop (descending)
        sorted_features = sorted(
            ablations.items(),
            key=lambda x: x[1]['roc_auc_drop'],
            reverse=True
        )
        
        for feature, data in sorted_features:
            print(f"  {feature:35s} | ROC-AUC drop: {data['roc_auc_drop']:+.4f} | "
                  f"Accuracy drop: {data['accuracy_drop']:+.4f}")
    
    if 'lightgbm' in results:
        print("\nLightGBM - Feature Importance (by ROC-AUC drop):")
        lgbm_results = results['lightgbm']
        ablations = lgbm_results['ablations']
        
        sorted_features = sorted(
            ablations.items(),
            key=lambda x: x[1]['roc_auc_drop'],
            reverse=True
        )
        
        for feature, data in sorted_features:
            print(f"  {feature:35s} | ROC-AUC drop: {data['roc_auc_drop']:+.4f} | "
                  f"Accuracy drop: {data['accuracy_drop']:+.4f}")


def main():
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
    
    # Run single feature ablation
    single_results = run_single_feature_ablation(df, model_type='both')
    
    # Run sequential ablation
    sequential_results = run_sequential_feature_ablation(df, model_type='both')
    
    # Combine results
    all_results = {**single_results, **sequential_results}
    
    # Print summary
    print_summary(single_results)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
