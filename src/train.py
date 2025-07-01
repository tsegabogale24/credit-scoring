"""credit_risk_training.py

End‑to‑end Credit Risk model training with MLflow tracking.
Run:

    python credit_risk_training.py --data ../data/processed/modeling_data.csv

"""

import argparse
from pathlib import Path
from datetime import datetime

import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "CreditRiskModeling"

MODELS = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "class_weight": [None, "balanced"],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        },
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "penalty": ["l2"],
            "class_weight": [None, "balanced"],
        },
    },
}


def load_and_preprocess(path: Path):
    """Load CSV and return numeric X, y."""
    df = pd.read_csv(path)
    if "is_high_risk" not in df.columns:
        raise KeyError("Column 'is_high_risk' not found in dataset.")

    # Drop ID‑like columns if present
    df = df.drop(columns=[c for c in ["CustomerId", "TransactionId"] if c in df.columns])

    y = df["is_high_risk"].astype(int)
    X = df.drop(columns=["is_high_risk"]).select_dtypes(include=["number"])

    if X.empty:
        raise ValueError("All non‑numeric features were dropped. No features left in X.")

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


import mlflow
from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(name, base_model, param_grid, X_train, y_train, X_test, y_test):
    """Train and log model with proper artifact path formatting"""
    with mlflow.start_run(nested=True, run_name=f"{name}_{datetime.now():%H%M%S}"):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)

        preds = gs.predict(X_test)
        probs = gs.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "best_cv_score": gs.best_score_,
        }

        # Corrected model logging - removed slash from artifact_path
        mlflow.sklearn.log_model(
            gs.best_estimator_,
            artifact_path=name.lower(),  # Changed from f"models/{name.lower()}"
            registered_model_name=f"CreditRisk_{name}",
        )
        
        # Log parameters and metrics
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics(metrics)

        return gs.best_estimator_, metrics


def main(data_path: Path):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    results = {}
    with mlflow.start_run(run_name="MainRun") as run:
        mlflow.log_params(
            {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "positive_class_ratio": y_train.mean(),
            }
        )

        for name, cfg in MODELS.items():
            print(f"Training {name}...")
            model, metrics = train_model(
                name,
                cfg["model"],
                cfg["params"],
                X_train,
                y_train,
                X_test,
                y_test,
            )
            results[name] = {"model": model, "metrics": metrics}
            print(f"{name} ROC‑AUC: {metrics['roc_auc']:.4f}")

        # choose best
        best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
        mlflow.set_tag("best_model", best_name)
        mlflow.log_metric("best_roc_auc", results[best_name]["metrics"]["roc_auc"])
        print(f"\nBest model: {best_name} (ROC‑AUC: {results[best_name]['metrics']['roc_auc']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to modeling_data.csv")
    args = parser.parse_args()
    main(args.data)
