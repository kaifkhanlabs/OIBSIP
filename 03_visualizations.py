import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
output_dir = os.path.join(root_dir, '4_screenshots')
os.makedirs(output_dir, exist_ok=True)

input_dir = os.path.join(root_dir, '3_outputs')
df = pd.read_csv(os.path.join(input_dir, 'cleaned_text.csv'))
df_pred = pd.read_csv(os.path.join(input_dir, 'predictions.csv'))

def save_plot(name):
    plt.savefig(os.path.join(output_dir, name), dpi=300, bbox_inches='tight')

plt.style.use('seaborn-v0_8-darkgrid')

print("Generating visualizations...")

plt.figure(figsize=(10, 6))
sentiment_counts = df['sentiment'].value_counts()
colors = {'positive': '#2ECC71', 'negative': '#E74C3C', 'neutral': '#F1C40F'}
bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
               color=[colors.get(x, '#95A5A6') for x in sentiment_counts.index])
plt.title('Sentiment Distribution in Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 5, f'{int(height)}', ha='center')
plt.tight_layout()
save_plot('sentiment_distribution.png')
plt.close()
print("✓ sentiment_distribution.png saved")

plt.figure(figsize=(12, 6))
word_count_pos = df[df['sentiment'] == 'positive']['word_count']
word_count_neg = df[df['sentiment'] == 'negative']['word_count']
word_count_neu = df[df['sentiment'] == 'neutral']['word_count']

plt.hist(word_count_pos, bins=20, alpha=0.5, label='Positive', color='#2ECC71', edgecolor='black')
plt.hist(word_count_neg, bins=20, alpha=0.5, label='Negative', color='#E74C3C', edgecolor='black')
plt.hist(word_count_neu, bins=20, alpha=0.5, label='Neutral', color='#F1C40F', edgecolor='black')
plt.title('Word Count Distribution by Sentiment', fontsize=16, fontweight='bold')
plt.xlabel('Number of Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_plot('word_count_distribution.png')
plt.close()
print("✓ word_count_distribution.png saved")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(df_pred['actual'], df_pred['predicted'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
save_plot('confusion_matrix.png')
plt.close()
print("✓ confusion_matrix.png saved")

plt.figure(figsize=(10, 6))
sentiment_pred = df_pred['predicted'].value_counts()
colors = {'positive': '#2ECC71', 'negative': '#E74C3C', 'neutral': '#F1C40F'}
bars = plt.bar(sentiment_pred.index, sentiment_pred.values,
               color=[colors.get(x, '#95A5A6') for x in sentiment_pred.index])
plt.title('Predicted Sentiment Distribution (Test Set)', fontsize=16, fontweight='bold')
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Number of Predictions', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{int(height)}', ha='center')
plt.tight_layout()
save_plot('predicted_distribution.png')
plt.close()
print("✓ predicted_distribution.png saved")

plt.figure(figsize=(12, 6))
df['length_category'] = pd.cut(df['word_count'], bins=[0, 3, 6, 20], labels=['Short (1-3)', 'Medium (4-6)', 'Long (7+)'])
length_sentiment = pd.crosstab(df['length_category'], df['sentiment'], normalize='index') * 100
length_sentiment.plot(kind='bar', figsize=(12, 6), color=['#E74C3C', '#F1C40F', '#2ECC71'])
plt.title('Sentiment Distribution by Tweet Length', fontsize=16, fontweight='bold')
plt.xlabel('Tweet Length', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Sentiment')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_plot('len_sentiment.png')
plt.close()
print("✓ len_sentiment.png saved")

def get_top_words(vectorizer, model, sentiment, n=10):
    class_idx = {'positive': 2, 'negative': 0, 'neutral': 1}[sentiment]
    feature_names = vectorizer.get_feature_names_out()
    coeffs = model.feature_log_prob_[class_idx]
    top_indices = np.argsort(coeffs)[-n:][::-1]
    return [(feature_names[i], coeffs[i]) for i in top_indices]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X = df['cleaned_text']
y = df['sentiment']
mask = X.notna() & y.notna()
X = X[mask].fillna('')
y = y[mask]
vectorizer = CountVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X_vec, y)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, sentiment in enumerate(['negative', 'neutral', 'positive']):
    top_words = get_top_words(vectorizer, model, sentiment, 10)
    words, scores = zip(*top_words)
    color = '#E74C3C' if sentiment == 'negative' else '#F1C40F' if sentiment == 'neutral' else '#2ECC71'
    axes[i].barh(words, scores, color=color)
    axes[i].set_title(f'Top Words - {sentiment.capitalize()}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Log Probability')
plt.tight_layout()
save_plot('top_words.png')
plt.close()
print("✓ top_words.png saved")

print("\nAll visualizations saved to 4_screenshots/")