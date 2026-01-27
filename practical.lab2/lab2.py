import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

sns.set_theme(style="whitegrid")

def run_pipeline():
    data_path = 'd:/Medical/mlmed2026/lab2/training_set_pixel_size_and_HC.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='pixel size(mm)', y='head circumference (mm)', alpha=0.6)
    plt.title('Pixel Size vs Head Circumference')
    plt.xlabel('Pixel Size (mm)')
    plt.ylabel('Head Circumference (mm)')
    plt.savefig('d:/Medical/mlmed2026/lab2/eda_scatter.png')
    print("\nSaved EDA scatter plot to 'eda_scatter.png'")

    plt.figure(figsize=(10, 6))
    sns.histplot(df['head circumference (mm)'], kde=True, bins=30)
    plt.title('Distribution of Head Circumference')
    plt.savefig('d:/Medical/mlmed2026/lab2/target_dist.png')
    print("Saved target distribution plot to 'target_dist.png'")


    X = df[['pixel size(mm)']]
    y = df['head circumference (mm)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance (Validation Set):")
    print(f"Mean Absolute Error (MAE): {mae:.4f} mm")
    print(f"R-squared (R2): {r2:.4f}")

    test_data_path = 'd:/Medical/mlmed2026/lab2/test_set_pixel_size.csv'
    if os.path.exists(test_data_path):
        test_df = pd.read_csv(test_data_path)
        print("\nTest Dataset Head:")
        print(test_df.head())

        test_X = test_df[['pixel size(mm)']]
        test_df['predicted_head_circumference'] = model.predict(test_X)
        test_df.to_csv('d:/Medical/mlmed2026/lab2/test_predictions.csv', index=False)
        print("\nSaved test predictions to 'test_predictions.csv'")

if __name__ == "__main__":
    run_pipeline()
