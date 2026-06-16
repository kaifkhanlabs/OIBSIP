import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('4_screenshots', exist_ok=True)

df = pd.read_csv('3_outputs/cleaned_retail_sales.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.day_name()

plt.style.use('seaborn-v0_8-darkgrid')

print("Generating visualizations...")

plt.figure(figsize=(12, 6))
category_sales = df.groupby('Product Category')['Total Amount'].sum().sort_values()
bars = plt.barh(category_sales.index, category_sales.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Total Sales by Product Category', fontsize=16, fontweight='bold')
plt.xlabel('Total Sales ($)', fontsize=12)
plt.ylabel('Product Category', fontsize=12)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 100, bar.get_y() + bar.get_height()/2, f'${width:,.0f}', va='center')
plt.tight_layout()
plt.savefig('4_screenshots/category_sales.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ category_sales.png saved")

plt.figure(figsize=(12, 6))
monthly_sales = df.groupby('Month')['Total Amount'].sum()
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', color='#FF6B6B', linewidth=2, markersize=8)
plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(range(1, 13))
for i, v in enumerate(monthly_sales.values):
    plt.text(i+1, v + 100, f'${v:,.0f}', ha='center', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4_screenshots/monthly_sales_trend.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ monthly_sales_trend.png saved")

plt.figure(figsize=(10, 6))
gender_sales = df.groupby('Gender')['Total Amount'].sum()
colors = ['#FF6B6B', '#4ECDC4']
plt.pie(gender_sales.values, labels=gender_sales.index, autopct='%1.1f%%', colors=colors, startangle=90, explode=(0.05, 0))
plt.title('Sales Distribution by Gender', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('4_screenshots/gender_sales.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ gender_sales.png saved")

plt.figure(figsize=(12, 6))
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_sales = df.groupby('Weekday')['Total Amount'].sum().reindex(weekday_order)
bars = plt.bar(weekday_sales.index, weekday_sales.values, color='#45B7D1', edgecolor='black')
plt.title('Sales by Day of Week', fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 100, f'${height:,.0f}', ha='center')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('4_screenshots/weekday_sales.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ weekday_sales.png saved")

plt.figure(figsize=(10, 6))
bins = [0, 18, 30, 45, 60, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '60+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
age_group_sales = df.groupby('Age Group', observed=False)['Total Amount'].sum()
bars = plt.bar(age_group_sales.index, age_group_sales.values, color='#FF6B6B', edgecolor='black')
plt.title('Sales by Age Group', fontsize=16, fontweight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 100, f'${height:,.0f}', ha='center')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('4_screenshots/age_group_sales.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ age_group_sales.png saved")

plt.figure(figsize=(10, 8))
correlation = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('4_screenshots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ correlation_heatmap.png saved")

plt.figure(figsize=(12, 6))
colors = {'Male': '#4ECDC4', 'Female': '#FF6B6B'}
for gender in df['Gender'].unique():
    subset = df[df['Gender'] == gender]
    plt.scatter(subset['Age'], subset['Total Amount'], alpha=0.5, label=gender, color=colors[gender])
plt.title('Age vs Total Spend by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Total Amount ($)', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('4_screenshots/age_vs_spend.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ age_vs_spend.png saved")

plt.figure(figsize=(12, 6))
category_gender = df.groupby(['Product Category', 'Gender'])['Total Amount'].sum().unstack().fillna(0)
category_gender.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
plt.title('Category Sales by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig('4_screenshots/category_gender.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ category_gender.png saved")

plt.figure(figsize=(12, 6))
avg_price_category = df.groupby('Product Category')['Price per Unit'].mean()
bars = plt.bar(avg_price_category.index, avg_price_category.values, color='#45B7D1', edgecolor='black')
plt.title('Average Price per Unit by Category', fontsize=16, fontweight='bold')
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Average Price ($)', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'${height:.0f}', ha='center')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('4_screenshots/avg_price_category.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ avg_price_category.png saved")

plt.figure(figsize=(12, 6))
category_quantity = df.groupby('Product Category')['Quantity'].sum()
bars = plt.bar(category_quantity.index, category_quantity.values, color='#4ECDC4', edgecolor='black')
plt.title('Total Quantity Sold by Category', fontsize=16, fontweight='bold')
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Quantity Sold', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height:.0f}', ha='center')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('4_screenshots/quantity_by_category.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ quantity_by_category.png saved")

print("\nAll visualizations saved to 4_screenshots/ folder")