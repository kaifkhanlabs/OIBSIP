import pandas as pd
import numpy as np
import os

os.makedirs('../3_outputs', exist_ok=True)

df = pd.read_csv('../3_outputs/cleaned_online_retail.csv')

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalAmount': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("RFM Data First 10 Rows:")
print(rfm.head(10))

print("\n" + "=" * 60)
print("RFM STATISTICS")
print("=" * 60)
print(rfm[['Recency', 'Frequency', 'Monetary']].describe())

rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])

rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

def segment_customer(row):
    if row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
        return 'Champions'
    elif row['R_Score'] >= 3 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
        return 'Loyal Customers'
    elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 1:
        return 'Potential Loyalists'
    elif row['R_Score'] >= 1 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
        return 'Recent Customers'
    elif row['R_Score'] >= 3 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
        return 'Promising'
    elif row['R_Score'] >= 1 and row['F_Score'] >= 1 and row['M_Score'] >= 1:
        return 'Need Attention'
    elif row['R_Score'] <= 2 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
        return 'About To Sleep'
    elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] >= 2:
        return 'At Risk'
    elif row['R_Score'] <= 2 and row['F_Score'] >= 3 and row['M_Score'] <= 2:
        return 'Cant Lose'
    elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
        return 'Lost'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

print("\n" + "=" * 60)
print("SEGMENT DISTRIBUTION")
print("=" * 60)
segment_counts = rfm['Segment'].value_counts()
print(segment_counts)
print("\nPercentage:")
print((segment_counts / len(rfm) * 100).round(2))

print("\n" + "=" * 60)
print("SEGMENT AVERAGE RFM VALUES")
print("=" * 60)
print(rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2))

rfm.to_csv('../3_outputs/rfm_data.csv', index=False)

print("\nRFM analysis complete. File saved to 3_outputs/rfm_data.csv")