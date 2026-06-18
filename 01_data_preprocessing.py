import pandas as pd
import numpy as np
import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))

os.makedirs(os.path.join(root_dir, '3_outputs'), exist_ok=True)
os.makedirs(os.path.join(root_dir, '4_screenshots'), exist_ok=True)

data_path = os.path.join(root_dir, '1_data', 'Twitter_Data.csv')
df = pd.read_csv(data_path)

print("=" * 70)
print("TWITTER SENTIMENT DATA PREPROCESSING")
print("=" * 70)
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print(f"\nSentiment Distribution:")
print(df['category'].value_counts())

sentiment_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
df['sentiment'] = df['category'].map(sentiment_map)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['clean_text'].apply(clean_text)

print("\n" + "=" * 70)
print("SAMPLE CLEANED TEXT")
print("=" * 70)
for i in range(5):
    print(f"Original: {str(df['clean_text'][i])[:80]}...")
    print(f"Cleaned: {df['cleaned_text'][i][:80]}...")
    print("-" * 40)

df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
df['char_count'] = df['cleaned_text'].apply(len)

print("\n" + "=" * 70)
print("TEXT STATISTICS")
print("=" * 70)
print(f"Average Word Count: {df['word_count'].mean():.2f}")
print(f"Average Character Count: {df['char_count'].mean():.2f}")

output_path = os.path.join(root_dir, '3_outputs', 'cleaned_text.csv')
df.to_csv(output_path, index=False)

print("\nPreprocessing complete. File saved to 3_outputs/cleaned_text.csv")