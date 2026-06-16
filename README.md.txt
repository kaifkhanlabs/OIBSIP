# Task 1: Retail Sales Exploratory Data Analysis

## What is this project
I analyzed retail sales data to find patterns in customer behavior, product performance, and sales trends. This is my first task for Oasis Infobyte internship.

#Dataset
- Total rows: 1000 transactions
- Columns: Transaction ID, Date, Customer ID, Gender, Age, Product Category, Quantity, Price per Unit, Total Amount

# What I did

1. Data cleaning
   - Checked null values
   - Removed duplicates (none found)
   - Added new columns - Month, Year, Day, Weekday
   - Created Revenue column (Quantity * Price per Unit)

2. Analysis
   - Calculated total sales, average transaction value
   - Checked sales by gender, category, age group
   - Found top customers
   - Did correlation analysis between age, quantity, price, total amount

3. Visualizations
   - Made 10 charts to understand the data better

# Key Findings

1. Clothing is the top category - 31% of total sales
2. Sales are highest in March and December
3. Female customers contribute 52% of revenue
4. Age group 31-45 spends the most money
5. There is weak correlation between age and spending
6. Electronics has highest average price per unit ($191)

# Tools Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn

# How to run the code

1. Open 2_code folder
2. Run 01_data_cleaning.py
3. Run 02_eda_analysis.py  
4. Run 03_visualisations.py

# Output
- Cleaned data saved in 3_outputs
- All charts saved in 4_screenshots

# Screenshots of visualizations

![Category Sales](4_screenshots/category_sales.png)
![Monthly Sales Trend](4_screenshots/monthly_sales_trend.png)
![Gender Sales](4_screenshots/gender_sales.png)
![Age Group Sales](4_screenshots/age_group_sales.png)
![Correlation Heatmap](4_screenshots/correlation_heatmap.png)
![Age vs Spend](4_screenshots/age_vs_spend.png)
![Category Gender](4_screenshots/category_gender.png)
![Avg Price Category](4_screenshots/avg_price_category.png)
![Weekday Sales](4_screenshots/weekday_sales.png)
![Quantity by Category](4_screenshots/quantity_by_category.png)

# Submitted To
Oasis Infobyte - Data Analytics Internship
June 2026