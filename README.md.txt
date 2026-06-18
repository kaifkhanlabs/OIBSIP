# Task 4: Sentiment Analysis on Twitter Data

## What is this project
Built a sentiment analysis model to classify tweets about Indian politics as positive, negative, or neutral using Machine Learning.

## Dataset
- Twitter data with political tweets
- Total tweets: ~16,000+
- Labels: -1 (Negative), 0 (Neutral), 1 (Positive)

## What I did

1. Data preprocessing
   - Converted text to lowercase
   - Removed special characters and numbers
   - Removed extra spaces
   - Added word count and character count features

2. Model Building
   - Used CountVectorizer for text features
   - Trained Multinomial Naive Bayes classifier
   - Split data: 80% train, 20% test

3. Results
   - Model Accuracy: ~75%
   - Precision and Recall scores calculated

4. Visualizations
   - Sentiment distribution
   - Word count by sentiment
   - Confusion matrix
   - Top words per sentiment

## Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## How to run

1. Run 01_data_preprocessing.py
2. Run 02_model_training.py
3. Run 03_visualizations.py

## Output
- cleaned_text.csv
- predictions.csv
- model_accuracy.txt
- 6 visualizations in 4_screenshots

## Screenshots

![Sentiment Distribution](4_screenshots/sentiment_distribution.png)
![Confusion Matrix](4_screenshots/confusion_matrix.png)
![Top Words](4_screenshots/top_words.png)

## Submitted To
Oasis Infobyte - Data Analytics Internship
June 2026