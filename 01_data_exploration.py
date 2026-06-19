import pandas as pd
import numpy as np
from pathlib import Path
import io

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
input_path = root_dir / '1_data' / 'credit_card_fraud.csv'
output_dir = root_dir / '3_outputs'
screenshot_dir = root_dir / '4_screenshots'

output_dir.mkdir(exist_ok=True)
screenshot_dir.mkdir(exist_ok=True)

with input_path.open('r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]
cleaned = []
for line in lines:
    line = line.lstrip('\ufeff')
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    cleaned.append(line)
df = pd.read_csv(io.StringIO('\n'.join(cleaned)))

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - DATA EXPLORATION")
print("=" * 70)
print(f"Dataset Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:")
print(df.dtypes)

print("\n" + "=" * 70)
print("FRAUD DISTRIBUTION")
print("=" * 70)
fraud_counts = df['is_fraud'].value_counts()
print(fraud_counts)
print(f"\nFraud Percentage: {fraud_counts[1] / len(df) * 100:.2f}%")

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)
print(df[['amount', 'previous_transactions', 'avg_transaction_amount', 'transaction_frequency']].describe())

print("\n" + "=" * 70)
print("FRAUD VS LEGIT - AVERAGE VALUES")
print("=" * 70)
fraud_avg = df.groupby('is_fraud').agg({
    'amount': 'mean',
    'previous_transactions': 'mean',
    'avg_transaction_amount': 'mean',
    'transaction_frequency': 'mean'
}).round(2)
print(fraud_avg)

print("\n" + "=" * 70)
print("TRANSACTION TYPE ANALYSIS")
print("=" * 70)
type_fraud = df.groupby(['transaction_type', 'is_fraud']).size().unstack(fill_value=0)
print(type_fraud)

print("\n" + "=" * 70)
print("MERCHANT CATEGORY ANALYSIS")
print("=" * 70)
category_fraud = df.groupby(['merchant_category', 'is_fraud']).size().unstack(fill_value=0)
print(category_fraud)

print("\n" + "=" * 70)
print("DEVICE TYPE ANALYSIS")
print("=" * 70)
device_fraud = df.groupby(['device_type', 'is_fraud']).size().unstack(fill_value=0)
print(device_fraud)

df.to_csv('../3_outputs/fraud_data_clean.csv', index=False)

print("\nData exploration complete. File saved to 3_outputs/fraud_data_clean.csv")