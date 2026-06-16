import pandas as pd
import numpy as np
import os

os.makedirs('3_outputs', exist_ok=True)

df = pd.read_csv('3_outputs/cleaned_retail_sales.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.day_name()

print("=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

total_sales = df['Total Amount'].sum()
total_transactions = df.shape[0]
avg_transaction = df['Total Amount'].mean()
avg_age = df['Age'].mean()
total_revenue = (df['Quantity'] * df['Price per Unit']).sum()

print(f"Total Sales: ${total_sales:,.2f}")
print(f"Total Revenue (Qty * Price): ${total_revenue:,.2f}")
print(f"Total Transactions: {total_transactions}")
print(f"Average Transaction Value: ${avg_transaction:.2f}")
print(f"Average Customer Age: {avg_age:.1f} years")

print("\n" + "=" * 60)
print("GENDER ANALYSIS")
print("=" * 60)

gender_sales = df.groupby('Gender')['Total Amount'].sum()
gender_count = df.groupby('Gender')['Customer ID'].nunique()
gender_transactions = df.groupby('Gender')['Transaction ID'].count()

print("Sales by Gender:")
print(gender_sales)
print("\nCustomer Count by Gender:")
print(gender_count)
print("\nTransactions by Gender:")
print(gender_transactions)

print("\n" + "=" * 60)
print("CATEGORY ANALYSIS")
print("=" * 60)

category_sales = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)
category_quantity = df.groupby('Product Category')['Quantity'].sum().sort_values(ascending=False)
category_transactions = df.groupby('Product Category')['Transaction ID'].count().sort_values(ascending=False)

print("Sales by Category:")
print(category_sales)
print("\nQuantity Sold by Category:")
print(category_quantity)
print("\nTransactions by Category:")
print(category_transactions)

print("\n" + "=" * 60)
print("MONTHLY ANALYSIS")
print("=" * 60)

monthly_sales = df.groupby('Month')['Total Amount'].sum()
monthly_orders = df.groupby('Month')['Transaction ID'].count()

print("Monthly Sales:")
for month, sales in monthly_sales.items():
    month_name = pd.to_datetime(f'2023-{month}-01').strftime('%B')
    print(f"{month_name}: ${sales:,.2f}")

print("\n" + "=" * 60)
print("AGE GROUP ANALYSIS")
print("=" * 60)

bins = [0, 18, 30, 45, 60, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '60+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

age_group_sales = df.groupby('Age Group', observed=False)['Total Amount'].sum()
age_group_count = df.groupby('Age Group', observed=False)['Customer ID'].nunique()
age_group_avg = df.groupby('Age Group', observed=False)['Total Amount'].mean()

print("Sales by Age Group:")
for group in age_group_sales.index:
    print(f"{group}: ${age_group_sales[group]:,.2f}")
print("\nCustomer Count by Age Group:")
print(age_group_count)
print("\nAverage Spend by Age Group:")
print(age_group_avg)

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

correlation = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].corr()
print(correlation)

print("\n" + "=" * 60)
print("TOP CUSTOMERS")
print("=" * 60)

top_customers = df.groupby('Customer ID')['Total Amount'].sum().sort_values(ascending=False).head(10)
top_frequency = df.groupby('Customer ID')['Transaction ID'].count().sort_values(ascending=False).head(10)

print("Top 10 Customers by Spend:")
for cust, amount in top_customers.items():
    print(f"{cust}: ${amount:,.2f}")

print("\nTop 10 Customers by Frequency:")
for cust, count in top_frequency.items():
    print(f"{cust}: {count} transactions")

print("\n" + "=" * 60)
print("CATEGORY x GENDER")
print("=" * 60)

category_gender = df.groupby(['Product Category', 'Gender'])['Total Amount'].sum().unstack().fillna(0)
print(category_gender)

print("\n" + "=" * 60)
print("WEEKDAY ANALYSIS")
print("=" * 60)

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_sales = df.groupby('Weekday')['Total Amount'].sum().reindex(weekday_order)

for day, sales in weekday_sales.items():
    print(f"{day}: ${sales:,.2f}")

df.to_csv('3_outputs/eda_retail_sales.csv', index=False)
print("\nEDA complete. Analysis saved to 3_outputs/eda_retail_sales.csv")