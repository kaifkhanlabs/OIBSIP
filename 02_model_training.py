import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))

os.makedirs(os.path.join(root_dir, '3_outputs'), exist_ok=True)

df = pd.read_csv(os.path.join(root_dir, '3_outputs', 'cleaned_text.csv'))

df = df.dropna(subset=['cleaned_text', 'sentiment'])

X = df['cleaned_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 70)
print("SENTIMENT ANALYSIS MODEL TRAINING")
print("=" * 70)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

vectorizer = CountVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("\nVocabulary size:", len(vectorizer.get_feature_names_out()))

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred))

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print(confusion_matrix(y_test, y_pred))

df_test = pd.DataFrame({
    'review': X_test,
    'actual': y_test,
    'predicted': y_pred
})

df_test.to_csv(os.path.join(root_dir, '3_outputs', 'predictions.csv'), index=False)

with open(os.path.join(root_dir, '3_outputs', 'model_accuracy.txt'), 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("SENTIMENT ANALYSIS MODEL RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("\nModel training complete. Files saved to 3_outputs/")