# In target_engineering.py
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_rfm(df, snapshot_date=None, customer_id_col='CustomerId', 
                 date_col='TransactionStartTime', amount_col='Amount'):
    """Enhanced RFM calculation with dynamic snapshot date"""
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + timedelta(days=1)
    
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        customer_id_col: 'count',
        amount_col: ['sum', 'mean']
    })
    rfm.columns = ['recency', 'frequency', 'monetary_total', 'monetary_avg']
    return rfm

def create_risk_labels(rfm_df, n_clusters=4, plot=True):
    """Enhanced risk labeling with visualization"""
    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    
    # Cluster with KMeans++
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify risk clusters
    cluster_stats = rfm_df.groupby('cluster').mean()
    rfm_df['is_high_risk'] = (
        (rfm_df['cluster'] == cluster_stats['recency'].idxmax()) |  # High recency
        (rfm_df['cluster'] == cluster_stats['monetary_total'].idxmin())  # Low spend
    ).astype(int)
    
    if plot:
        plot_rfm_clusters(rfm_df)
    
    return rfm_df[['cluster', 'is_high_risk']]

def plot_rfm_clusters(rfm_df):
    """Enhanced cluster visualization"""
    # Select only the metrics we want to plot (exclude monetary_avg and cluster/is_high_risk if present)
    metrics = ['recency', 'frequency', 'monetary_total']
    
    # Create the right number of subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    
    # If there's only one metric, axes won't be an array
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        sns.boxplot(x='cluster', y=metric, data=rfm_df, ax=axes[i])
        axes[i].set_title(f'{metric.capitalize()} by Cluster')
    plt.tight_layout()
    plt.show()

def create_final_dataset(processed_dir='../data/processed'):
    """Complete dataset assembly with validation"""
    processed_dir = Path(processed_dir)
    
    # Load components
    features = pd.read_csv(processed_dir/'X_preprocessed.csv')
    customer_ids = pd.read_csv(processed_dir/'customer_ids.csv')
    targets = pd.read_csv(processed_dir/'targets.csv')
    
    # Validate shapes
    assert len(features) == len(customer_ids) == len(targets), "Mismatched row counts"
    
    # Merge final dataset
    final_data = pd.concat([customer_ids, features, targets], axis=1)
    final_data.to_csv(processed_dir/'modeling_data.csv', index=False)
    return final_data