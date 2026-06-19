# Task 5: Credit Card Fraud Detection

## What is this project
Built a machine learning model to detect fraudulent credit card transactions using classification algorithms.

## Dataset
- 100 transactions
- Features: amount, transaction_type, merchant_category, time_of_day, day_of_week, location, device_type, previous_transactions, avg_transaction_amount, transaction_frequency
- Target: is_fraud (0 = Legit, 1 = Fraud)

## What I did

1. Data Exploration
   - Checked fraud distribution (20% fraud, 80% legit)
   - Analyzed patterns by category, device, time

2. Model Building
   - Tried 3 models: Logistic Regression, Decision Tree, Random Forest
   - Random Forest performed best

3. Results
   - Accuracy: ~95%
   - Precision: ~95%
   - Recall: ~95%
   - F1-Score: ~95%

4. Key Insights
   - Electronics and Luxury categories have highest fraud
   - Night transactions are more risky
   - Desktop devices show more fraud patterns
   - High amount transactions are suspicious

## Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## How to run

1. Run 01_data_exploration.py
2. Run 02_model_training.py
3. Run 03_visualizations.py

## Output
- fraud_data_clean.csv
- predictions.csv
- model_results.txt
- 8 visualizations in 4_screenshots

## Screenshots

![Fraud Distribution](4_screenshots/fraud_distribution.png)
![Fraud by Category](4_screenshots/fraud_by_category.png)
![Confusion Matrix](4_screenshots/confusion_matrix.png)
![Feature Importance](4_screenshots/feature_importance.png)

## Submitted To
Oasis Infobyte - Data Analytics Internship
June 2026