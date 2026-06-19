import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
import os

os.makedirs('../3_outputs', exist_ok=True)

df = pd.read_csv('../3_outputs/fraud_data_clean.csv')

features = ['amount', 'transaction_type', 'merchant_category', 'time_of_day', 
            'day_of_week', 'location', 'device_type', 'previous_transactions', 
            'avg_transaction_amount', 'transaction_frequency']
target = 'is_fraud'

df_encoded = df.copy()
label_encoders = {}

for col in ['transaction_type', 'merchant_category', 'time_of_day', 
            'day_of_week', 'location', 'device_type']:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X = df_encoded[features]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 70)
print("FRAUD DETECTION MODEL TRAINING")
print("=" * 70)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Fraud cases in training: {y_train.sum()}")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {results[name]['accuracy']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall: {results[name]['recall']:.4f}")
    print(f"  F1-Score: {results[name]['f1']:.4f}")

print("\n" + "=" * 70)
print("BEST MODEL: RANDOM FOREST")
print("=" * 70)

best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_best):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_best):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

df_test = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_best
})
df_test.to_csv('../3_outputs/predictions.csv', index=False)

with open('../3_outputs/model_results.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("FRAUD DETECTION MODEL RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Best Model: Random Forest\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}\n")
    f.write(f"Precision: {precision_score(y_test, y_pred_best):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred_best):.4f}\n")
    f.write(f"F1-Score: {f1_score(y_test, y_pred_best):.4f}\n\n")
    f.write("Feature Importance:\n")
    f.write(feature_importance.to_string())

print("\nModel training complete. Results saved to 3_outputs/model_results.txt")