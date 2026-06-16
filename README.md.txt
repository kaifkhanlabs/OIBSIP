# Task 2: Customer Segmentation Analysis

## What is this project
I analyzed customer purchase data to group customers into segments based on their buying behavior. This helps businesses target different customer groups with personalized marketing.

## Dataset
- Online Retail dataset (UK-based e-commerce)
- Total transactions: ~500,000 rows
- Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

## What I did

1. Data cleaning
   - Removed rows with missing CustomerID
   - Removed negative quantities (returns)
   - Removed zero or negative prices
   - Converted InvoiceDate to datetime

2. RFM Analysis
   - Recency: Days since last purchase
   - Frequency: Total number of orders
   - Monetary: Total money spent
   - Created RFM scores (1-4)
   - Assigned segments (Champions, Loyal, At Risk, etc.)

3. K-Means Clustering
   - Scaled RFM features
   - Used elbow method to find optimal k=4
   - Created 4 customer clusters

## Key Findings

1. 4 customer clusters identified:
   - Cluster 0: High value, frequent buyers
   - Cluster 1: Low value, recent buyers
   - Cluster 2: Medium value, occasional buyers
   - Cluster 3: High recency, low frequency

2. RFM segments distribution:
   - Champions: Highest value customers
   - Need Attention: Most customers
   - Lost: Long time no purchase

3. Top 3 customer segments by monetary value:
   - Champions
   - Loyal Customers
   - Potential Loyalists

## Tools Used
- Python
- Pandas, NumPy
- Scikit-learn (KMeans, PCA)
- Matplotlib, Seaborn

## How to run the code

1. Open 2_code folder
2. Run 01_data_cleaning.py
3. Run 02_customer_rfm.py
4. Run 03_kmeans_clustering.py
5. Run 04_visualizations.py

## Output
- Cleaned data saved in 3_outputs
- RFM data with segments
- Clustered customer data
- 6 visualizations in 4_screenshots



## Submitted To
Oasis Infobyte - Data Analytics Internship
June 2026