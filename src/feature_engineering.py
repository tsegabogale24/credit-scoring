import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Robust outlier handling using IQR method"""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.feature_ranges = {}
        
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.feature_ranges[col] = (q1 - self.factor*iqr, q3 + self.factor*iqr)
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        for col, (lower, upper) in self.feature_ranges.items():
            X[col] = X[col].clip(lower, upper)
        return X.values
    
    def get_feature_names_out(self, input_features=None):
        return input_features

def extract_datetime_features(df, timestamp_col):
    """Enhanced datetime feature extraction"""
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["transaction_weekday"] = df[timestamp_col].dt.weekday
    for unit in ['hour', 'day', 'month', 'year']:
        df[f"transaction_{unit}"] = getattr(df[timestamp_col].dt, unit)
    return df

def aggregate_customer_features(df):
    """Comprehensive customer aggregations"""
    agg_df = df.groupby("CustomerId").agg({
        "Amount": ["sum", "mean", "std", "count", "max", "min"],
        "Value": ["mean", "std"]
    })
    agg_df.columns = [
        "total_amount", "avg_amount", "std_amount", "transaction_count",
        "max_amount", "min_amount", "avg_value", "std_value"
    ]
    return agg_df.reset_index()

def build_feature_pipeline(numerical_features, categorical_features):
    """Complete preprocessing pipeline with outlier handling"""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier", OutlierRemover()),  # Integrated outlier handling
        ("scaler", RobustScaler())      # Robust scaling after outlier treatment
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return Pipeline([("preprocessing", preprocessor)])