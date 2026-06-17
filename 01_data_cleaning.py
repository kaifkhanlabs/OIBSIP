import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
input_path = os.path.join(project_root, '1_data', 'AB_NYC_2019.csv')
output_dir = os.path.join(project_root, '3_outputs')
screenshot_dir = os.path.join(project_root, '4_screenshots')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(screenshot_dir, exist_ok=True)

df = pd.read_csv(input_path)

print("=" * 70)
print("INITIAL DATA EXPLORATION")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")

df_clean = df.copy()

df_clean['last_review'] = pd.to_datetime(df_clean['last_review'], errors='coerce')

df_clean = df_clean.dropna(subset=['name', 'host_name'])

df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)

df_clean['last_review'] = df_clean['last_review'].fillna(pd.to_datetime('2019-01-01'))

print(f"\nMissing values after cleaning:")
print(df_clean.isnull().sum())

df_clean = df_clean[df_clean['price'] > 0]
df_clean = df_clean[df_clean['price'] < 1000]

df_clean = df_clean[df_clean['minimum_nights'] < 365]
df_clean = df_clean[df_clean['availability_365'] <= 365]

print(f"\nShape after cleaning: {df_clean.shape}")

df_clean.to_csv(os.path.join(output_dir, 'cleaned_airbnb.csv'), index=False)

print("\nData cleaning complete. File saved to 3_outputs/cleaned_airbnb.csv")