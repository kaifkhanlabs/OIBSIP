import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
output_dir = root_dir / '4_screenshots'
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(root_dir / '3_outputs' / 'fraud_data_clean.csv')

try:
    df_pred = pd.read_csv(root_dir / '3_outputs' / 'predictions.csv')
except FileNotFoundError:
    df_pred = None
    print('! predictions.csv not found; confusion_matrix and feature_importance plots will be skipped')

plt.style.use('seaborn-v0_8-darkgrid')

print("Generating visualizations...")

plt.figure(figsize=(10, 6))
fraud_counts = df['is_fraud'].value_counts()
colors = ['#2ECC71', '#E74C3C']
labels = ['Legit', 'Fraud']
bars = plt.bar(labels, fraud_counts.values, color=colors, edgecolor='black')
plt.title('Fraud vs Legit Transactions', fontsize=16, fontweight='bold')
plt.xlabel('Transaction Type', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'{int(height)}', ha='center')
plt.tight_layout()
plt.savefig(output_dir / 'fraud_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("fraud_distribution.png saved")

plt.figure(figsize=(12, 6))
fraud_by_category = df.groupby('merchant_category')['is_fraud'].sum().sort_values()
bars = plt.barh(fraud_by_category.index, fraud_by_category.values, color='#E74C3C', edgecolor='black')
plt.title('Fraud Count by Merchant Category', fontsize=16, fontweight='bold')
plt.xlabel('Number of Fraud Transactions', fontsize=12)
plt.ylabel('Merchant Category', fontsize=12)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width)}', va='center')
plt.tight_layout()
plt.savefig(output_dir / 'fraud_by_category.png', dpi=300, bbox_inches='tight')
plt.close()
print("fraud_by_category.png saved")

plt.figure(figsize=(12, 6))
fraud_by_device = df.groupby('device_type')['is_fraud'].sum()
bars = plt.bar(fraud_by_device.index, fraud_by_device.values, color='#E74C3C', edgecolor='black')
plt.title('Fraud Count by Device Type', fontsize=16, fontweight='bold')
plt.xlabel('Device Type', fontsize=12)
plt.ylabel('Number of Fraud Transactions', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}', ha='center')
plt.tight_layout()
plt.savefig(output_dir / 'fraud_by_device.png', dpi=300, bbox_inches='tight')
plt.close()
print("fraud_by_device.png saved")

plt.figure(figsize=(12, 6))
fraud_by_type = df.groupby('transaction_type')['is_fraud'].sum()
bars = plt.bar(fraud_by_type.index, fraud_by_type.values, color='#E74C3C', edgecolor='black')
plt.title('Fraud Count by Transaction Type', fontsize=16, fontweight='bold')
plt.xlabel('Transaction Type', fontsize=12)
plt.ylabel('Number of Fraud Transactions', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}', ha='center')
plt.tight_layout()
plt.savefig(output_dir / 'fraud_by_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("fraud_by_type.png saved")

plt.figure(figsize=(12, 6))
fraud_by_time = df.groupby('time_of_day')['is_fraud'].sum()
bars = plt.bar(fraud_by_time.index, fraud_by_time.values, color='#E74C3C', edgecolor='black')
plt.title('Fraud Count by Time of Day', fontsize=16, fontweight='bold')
plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Number of Fraud Transactions', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}', ha='center')
plt.tight_layout()
plt.savefig(output_dir / 'fraud_by_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("fraud_by_time.png saved")

plt.figure(figsize=(12, 6))
fraud_legit_avg = df.groupby('is_fraud')[['amount', 'previous_transactions', 'avg_transaction_amount', 'transaction_frequency']].mean().T
fraud_legit_avg.columns = ['Legit', 'Fraud']
fraud_legit_avg.plot(kind='bar', figsize=(12, 6), color=['#2ECC71', '#E74C3C'], edgecolor='black')
plt.title('Average Values: Fraud vs Legit', fontsize=16, fontweight='bold')
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Average Value', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Transaction Type')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'fraud_legit_avg.png', dpi=300, bbox_inches='tight')
plt.close()
print("fraud_legit_avg.png saved")

if df_pred is not None:
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(df_pred['actual'], df_pred['predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix - Random Forest', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("confusion_matrix.png saved")
else:
    print("! confusion_matrix.png could not be generated because predictions.csv is missing")

try:
    feature_importance = pd.read_csv(root_dir / '3_outputs' / 'model_results.txt', skiprows=13, delimiter=' ', header=None)
    feature_importance.columns = ['feature', 'importance']
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_importance['feature'], feature_importance['importance'], color='#3498DB', edgecolor='black')
    plt.title('Feature Importance - Random Forest', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("feature_importance.png saved")
except:
    print("! feature_importance.png could not be generated (run model first)")

print("\nAll visualizations saved to 4_screenshots/")