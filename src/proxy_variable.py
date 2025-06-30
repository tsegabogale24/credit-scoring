import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_rfm(data, customer_id_col='CustomerId', 
                 date_col='TransactionStartTime', 
                 amount_col='Amount'):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
    
    Args:
        data: Raw transaction DataFrame
        customer_id_col: Name of customer ID column
        date_col: Name of transaction date column
        amount_col: Name of transaction amount column
    
    Returns:
        DataFrame with RFM values indexed by CustomerId
    """
    # Ensure proper data types
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Set snapshot date
    snapshot_date = data[date_col].max() + timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = data.groupby(customer_id_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,  # Recency
        customer_id_col: 'count',                            # Frequency
        amount_col: 'sum'                                    # Monetary
    }).rename(columns={
        date_col: 'recency',
        customer_id_col: 'frequency',
        amount_col: 'monetary'
    })
    
    return rfm

def create_high_risk_labels(rfm_df, n_clusters=3, plot=False, random_state=42):
    """
    Create high-risk labels using K-Means clustering on RFM metrics.
    
    Args:
        rfm_df: DataFrame with recency, frequency, monetary columns
        n_clusters: Number of clusters to create
        plot: Whether to show cluster visualizations
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with cluster assignments and is_high_risk column
    """
    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
    
    # Cluster customers
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster (lowest monetary value)
    cluster_means = rfm_df.groupby('cluster').mean()
    high_risk_cluster = cluster_means['monetary'].idxmin()
    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
    
    if plot:
        plot_cluster_characteristics(rfm_df)
    
    return rfm_df.reset_index()  # Ensure CustomerId is a column

def plot_cluster_characteristics(rfm_df):
    """Visualize RFM characteristics by cluster."""
    plt.figure(figsize=(15, 5))
    metrics = ['recency', 'frequency', 'monetary']
    titles = ['Recency (Days)', 'Frequency', 'Monetary Value']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x='cluster', y=metric, data=rfm_df)
        plt.title(f'{titles[i-1]} by Cluster')
    
    plt.tight_layout()
    plt.show()
    print(f"High-risk cluster: {rfm_df['cluster'].min()}")

def merge_target_with_features(main_df, rfm_df, customer_id_col='CustomerId'):
    """
    Safely merge target and RFM features into main dataframe.
    
    Args:
        main_df: Main DataFrame containing features
        rfm_df: DataFrame with RFM metrics and target variable
        customer_id_col: Name of customer ID column
    
    Returns:
        Merged DataFrame
    """
    # Ensure CustomerId is a column in both DataFrames
    if customer_id_col not in main_df.columns:
        main_df = main_df.reset_index()
    if customer_id_col not in rfm_df.columns:
        rfm_df = rfm_df.reset_index()
    
    return main_df.merge(
        rfm_df[[customer_id_col, 'recency', 'frequency', 'monetary', 'is_high_risk']],
        on=customer_id_col,
        how='left'
    )