# Task 3: NYC Airbnb Data Analysis

## What is this project

Analyzed New York City Airbnb data to understand pricing trends, neighborhood distribution, and room type popularity.

## Dataset

- Total rows: 48,895 (after cleaning)
- Columns: 16
- Key features: id, name, host_id, host_name, neighbourhood_group, neighbourhood, latitude, longitude, room_type, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, calculated_host_listings_count, availability_365

## What I did

1. Data cleaning
   - Converted last_review to datetime
   - Filled missing reviews_per_month with 0
   - Filled missing last_review with default date
   - Removed rows with missing name or host_name
   - Removed price outliers (>1000)
   - Removed minimum_nights outliers (>365)

2. Analysis
   - Calculated average price, median price
   - Analyzed price by neighborhood group
   - Analyzed price by room type
   - Found top neighborhoods and hosts

3. Visualizations
   - 8 charts showing key insights

## Key Findings

1. Manhattan has the highest average price ($196) and highest number of listings
2. Entire home/apt is the most common room type (52%)
3. Average price across NYC is $152
4. Most listings have high availability (avg 113 days/year)
5. Weak correlation between price and reviews

## Tools Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn

## How to run

1. Run 01_data_cleaning.py
2. Run 02_eda_analysis.py
3. Run 03_visualizations.py

## Output

- cleaned_airbnb.csv
- eda_report.txt
- summary_stats.csv
- 8 visualizations in 4_screenshots

## Screenshots

![Listings by Neighbourhood](4_screenshots/listings_by_neighbourhood.png)
![Avg Price by Neighbourhood](4_screenshots/avg_price_by_neighbourhood.png)
![Room Type Distribution](4_screenshots/room_type_pie.png)
![Price Distribution](4_screenshots/price_distribution.png)

## Submitted To

Oasis Infobyte - Data Analytics Internship
June 2026