import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------
# Feature Aggregation
# ------------------------
def aggregate_customer_features(df):
    """Aggregate transaction data at customer level"""
    agg_df = df.groupby("CustomerId").agg({
        "Amount": ["sum", "mean", "std", "count"]
    })
    agg_df.columns = [
        "total_amount", "avg_amount", "std_amount", "transaction_count"
    ]
    agg_df.reset_index(inplace=True)
    return agg_df

# ------------------------
# Datetime Feature Extraction
# ------------------------
def extract_datetime_features(df, timestamp_col):
    """Extract datetime features from timestamp column"""
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["transaction_hour"] = df[timestamp_col].dt.hour
    df["transaction_day"] = df[timestamp_col].dt.day
    df["transaction_month"] = df[timestamp_col].dt.month
    df["transaction_year"] = df[timestamp_col].dt.year
    return df

# ------------------------
# Outlier Handling
# ------------------------
class OutlierRemover(BaseEstimator, TransformerMixin):
    """Clip outliers using IQR method"""
    def __init__(self):
        self.q1 = None
        self.q3 = None
        self.iqr = None
        self._feature_names_in = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        return self

    def transform(self, X):
        X = np.asarray(X)
        lower = self.q1 - 1.5 * self.iqr
        upper = self.q3 + 1.5 * self.iqr
        return np.clip(X, lower, upper)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation"""
        if input_features is None:
            if self._feature_names_in is None:
                raise ValueError(
                    "Unable to generate feature names without input feature names"
                )
            input_features = self._feature_names_in
        return np.asarray(input_features, dtype=object)

# ------------------------
# Pipeline Builder
# ------------------------
def build_feature_pipeline(numerical_features, categorical_features):
    """Build preprocessing pipeline for features"""
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier", OutlierRemover()),
        ("scaler", RobustScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", 
                                sparse_output=False, 
                                drop="if_binary"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return Pipeline([("preprocessing", preprocessor)])