import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    print("=== EXPLORATORY DATA ANALYSIS STARTED ===\n")

    # Dataset Info
    print("### Dataset Info ###")
    df.info()

    # Head
    print("\n### First 5 Rows ###")
    df.head()

    # Summary Statistics (numerical only)
    print("\n### Summary Statistics ###")
    df.describe().T

    # Identify missing values
    print("\n### Missing Values ###")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Distribution of Numerical Features
    numerical_cols = ['Amount', 'Value']
    print("\n### Distribution of Numerical Features ###")
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=50, kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    # Distribution of Categorical Features (Top 6)
    categorical_cols = ['ProductCategory', 'ChannelId', 'CurrencyCode', 'PricingStrategy', 'FraudResult']
    print("\n### Distribution of Categorical Features ###")
    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index[:10], palette='viridis')
        plt.title(f'Frequency of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Correlation heatmap (only numerical features)
    print("\n### Correlation Heatmap ###")
    corr_matrix = df[numerical_cols + ['CountryCode', 'PricingStrategy', 'FraudResult']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Between Numerical Features")
    plt.tight_layout()
    plt.show()

    # Outlier Detection using Boxplots
    print("\n### Outlier Detection with Boxplots ###")
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], color='salmon')
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

    print("\n=== EDA Completed ===")
