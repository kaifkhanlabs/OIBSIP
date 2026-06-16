import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

os.makedirs('../3_outputs', exist_ok=True)

rfm = pd.read_csv('../3_outputs/rfm_data.csv')

features = rfm[['Recency', 'Frequency', 'Monetary']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

print("Inertia values for k=1 to 10:")
for i, val in enumerate(inertia, 1):
    print(f"k={i}: {val:.2f}")

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(scaled_features)

print("\n" + "=" * 60)
print("CLUSTER CENTERS (Scaled)")
print("=" * 60)
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=['Recency', 'Frequency', 'Monetary']
)
print(cluster_centers.round(2))

print("\n" + "=" * 60)
print("CLUSTER DISTRIBUTION")
print("=" * 60)
print(rfm['Cluster'].value_counts().sort_index())

print("\n" + "=" * 60)
print("CLUSTER AVERAGE VALUES")
print("=" * 60)
cluster_avg = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
print(cluster_avg)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
rfm['PCA1'] = pca_result[:, 0]
rfm['PCA2'] = pca_result[:, 1]

rfm.to_csv('../3_outputs/segmented_customers.csv', index=False)

print("\nK-Means clustering complete. File saved to 3_outputs/segmented_customers.csv")