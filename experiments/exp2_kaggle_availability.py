"""
experiments/exp2_kaggle_availability.py
Phase 2: Kaggle donor-availability classification experiment.
Predicts 'availability' based on demographic and historical data.
"""

import json
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay

from src.data.kaggle_loader import prepare_kaggle_data, get_preprocessor

def run_experiment():
    print("--- Starting Kaggle Donor-Availability Classification ---")
    data_path = "data/raw/blood_donor_dataset_modeling.csv"
    data = prepare_kaggle_data(data_path, seed=42)
    X = data['X']
    y = data['y']
    
    print(f"Loaded dataset: {X.shape[0]} rows.")
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    preprocessor = get_preprocessor()
    
    models = {
        'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), {
            'classifier__C': [0.1, 1.0, 10.0]
        }),
        'Random Forest': (RandomForestClassifier(random_state=42), {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 5, 10]
        }),
        'SVM': (SVC(probability=True, random_state=42), {
            'classifier__C': [0.1, 1.0, 10.0]
        }),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1]
        })
    }
    
    results = []
    cv_results_list = []
    best_model_name = None
    best_auc = -1
    best_pipeline = None
    
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    fig_cal, ax_cal = plt.subplots(figsize=(8, 6))
    
    confusion_matrices = {}
    
    for name, (clf, param_grid) in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=5, cv=5, scoring='roc_auc',
            random_state=42, n_jobs=-1
        )
        
        start_time = time.time()
        search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Cross-validation metrics
        cv_res = search.cv_results_
        best_idx = search.best_index_
        cv_roc_auc = cv_res['mean_test_score'][best_idx]
        cv_results_list.append({
            'Model': name,
            'CV_ROC_AUC_Mean': cv_roc_auc,
            'CV_ROC_AUC_Std': cv_res['std_test_score'][best_idx]
        })
        
        # Test evaluation
        start_time = time.time()
        y_pred = search.predict(X_test)
        y_prob = search.predict_proba(X_test)[:, 1]
        eval_time = time.time() - start_time
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        confusion_matrices[name] = cm.tolist()
        
        results.append({
            'Model': name,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'Brier_Score': brier,
            'Train_Time_s': train_time,
            'Eval_Time_s': eval_time
        })
        
        if cv_roc_auc > best_auc:
            best_auc = cv_roc_auc
            best_model_name = name
            best_pipeline = search.best_estimator_
            
        # Plotting
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_pr)
        CalibrationDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_cal)
        
    print(f"Best model selected: {best_model_name} (CV ROC-AUC: {best_auc:.4f})")
    
    # Save outputs
    out_dir = Path("reports")
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(results).to_csv(tables_dir / "availability_model_metrics.csv", index=False)
    pd.DataFrame(cv_results_list).to_csv(tables_dir / "availability_cv_metrics.csv", index=False)
    
    with open(tables_dir / "availability_confusion_matrices.json", "w") as f:
        json.dump(confusion_matrices, f, indent=4)
        
    ax_roc.set_title("ROC Curves - Availability Classification")
    fig_roc.savefig(figs_dir / "availability_roc_curves.png")
    
    ax_pr.set_title("Precision-Recall Curves - Availability Classification")
    fig_pr.savefig(figs_dir / "availability_precision_recall_curves.png")
    
    ax_cal.set_title("Calibration Plots - Availability Classification")
    fig_cal.savefig(figs_dir / "availability_calibration.png")
    
    # Save best model
    with open("models/best_availability_model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
        
    # Write report
    report_content = f"""# Kaggle Donor-Availability Classification Experiment Report

## Boundary Limitation
This experiment predicts the Kaggle dataset's `availability` label only. This is a general availability classification and does **not** represent a validated emergency recruitment-response probability. This probability must not be fed directly into the UDI field during emergency simulations unless properly calibrated against authorized invitation-response data. The UDI simulation will continue to use synthetic labels for its comparative experiments.

## Evaluation
The models were trained on 80% of the dataset using stratified 5-fold CV to select hyperparameters. They were then evaluated on a held-out 20% test set.

**Best Model Selected:** {best_model_name}
Selection was based on the highest mean ROC-AUC during 5-fold cross-validation on the training set (CV ROC-AUC: {best_auc:.4f}).

Metrics, cross-validation results, and figures are available in the `reports/` directory.
"""
    with open(out_dir / "availability_experiment_report.md", "w") as f:
        f.write(report_content)
        
    print("Experiment finished successfully.")

if __name__ == "__main__":
    run_experiment()
