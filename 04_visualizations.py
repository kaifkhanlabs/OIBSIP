import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../4_screenshots', exist_ok=True)

rfm = pd.read_csv('../3_outputs/segmented_customers.csv')

plt.style.use('seaborn-v0_8-darkgrid')

print("Generating visualizations...")

plt.figure(figsize=(12, 6))
segment_counts = rfm['Segment'].value_counts()
colors = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#1abc9c', '#34495e', '#95a5a6']
bars = plt.bar(segment_counts.index, segment_counts.values, color=colors[:len(segment_counts)])
plt.title('Customer Segments Distribution (RFM)', fontsize=16, fontweight='bold')
plt.xlabel('Segment', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 5, f'{int(height)}', ha='center')
plt.tight_layout()
plt.savefig('../4_screenshots/rfm_segments.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ rfm_segments.png saved")

plt.figure(figsize=(14, 6))
cluster_avg = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_avg.plot(kind='bar', figsize=(14, 6), color=['#e74c3c', '#3498db', '#2ecc71'])
plt.title('Cluster Average RFM Values', fontsize=16, fontweight='bold')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Average Value', fontsize=12)
plt.xticks(rotation=0)
plt.legend(['Recency (days)', 'Frequency (orders)', 'Monetary (spend)'])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../4_screenshots/cluster_rfm_values.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ cluster_rfm_values.png saved")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(rfm['PCA1'], rfm['PCA2'], c=rfm['Cluster'], cmap='viridis', alpha=0.6, s=30)
plt.colorbar(scatter, label='Cluster')
plt.title('Customer Clusters (PCA Projection)', fontsize=16, fontweight='bold')
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.tight_layout()
plt.savefig('../4_screenshots/cluster_pca.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ cluster_pca.png saved")

plt.figure(figsize=(10, 8))
cluster_summary = rfm.groupby('Cluster')[['Monetary', 'Frequency']].mean()
cluster_summary.plot(kind='scatter', x='Frequency', y='Monetary', s=200, c=['#e74c3c', '#3498db', '#2ecc71', '#f1c40f'])
plt.title('Cluster: Frequency vs Monetary', fontsize=16, fontweight='bold')
plt.xlabel('Average Frequency', fontsize=12)
plt.ylabel('Average Monetary ($)', fontsize=12)
for i, row in cluster_summary.iterrows():
    plt.annotate(f'Cluster {i}', (row['Frequency'] + 0.5, row['Monetary'] + 50), fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../4_screenshots/cluster_freq_monetary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ cluster_freq_monetary.png saved")

plt.figure(figsize=(12, 6))
rfm_group = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
bars = plt.bar(rfm_group.index, rfm_group.values, color='#3498db', edgecolor='black')
plt.title('Total Monetary Value by Segment', fontsize=16, fontweight='bold')
plt.xlabel('Segment', fontsize=12)
plt.ylabel('Total Monetary Value ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 50, f'${height:,.0f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('../4_screenshots/segment_monetary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ segment_monetary.png saved")

plt.figure(figsize=(12, 6))
rfm_scores = rfm[['R_Score', 'F_Score', 'M_Score']].mean()
bars = plt.bar(['Recency Score', 'Frequency Score', 'Monetary Score'], rfm_scores, color=['#e74c3c', '#3498db', '#2ecc71'])
plt.title('Average RFM Scores', fontsize=16, fontweight='bold')
plt.ylabel('Average Score (1-4)', fontsize=12)
plt.ylim(0, 4.5)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{height:.2f}', ha='center')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../4_screenshots/rfm_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ rfm_scores.png saved")

print("\nAll visualizations saved to 4_screenshots/")