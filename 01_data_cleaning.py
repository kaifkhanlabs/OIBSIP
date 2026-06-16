import pandas as pd
import numpy as np
import os

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

os.makedirs(os.path.join(project_root, '3_output'), exist_ok=True)

df = pd.read_excel(os.path.join(project_root, '1_data/Online Retail.xlsx'))

print("Original Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("UNIQUE VALUES COUNT")
print("=" * 60)
print(df.nunique())

df_clean = df.copy()

df_clean = df_clean.dropna(subset=['CustomerID'])

df_clean = df_clean[df_clean['Quantity'] > 0]
df_clean = df_clean[df_clean['UnitPrice'] > 0]

df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], format='%m/%d/%Y %H:%M', errors='coerce')

df_clean = df_clean.dropna(subset=['InvoiceDate'])

df_clean = df_clean.drop_duplicates()

df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

print("\n" + "=" * 60)
print("CLEANED DATA INFO")
print("=" * 60)
print("New Shape:", df_clean.shape)
print("\nMissing Values After Cleaning:")
print(df_clean.isnull().sum())

print("\nDate Range:")
print("Min:", df_clean['InvoiceDate'].min())
print("Max:", df_clean['InvoiceDate'].max())

output_path = os.path.join(project_root, '3_output/cleaned_online_retail.csv')
df_clean.to_csv(output_path, index=False)

print("\nData cleaning complete. File saved to 3_output/cleaned_online_retail.csv")