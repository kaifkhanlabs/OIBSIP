import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
output_dir = os.path.join(project_root, '3_outputs')

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(os.path.join(output_dir, 'cleaned_airbnb.csv'))

print("=" * 70)
print("NYC AIRBNB EDA")
print("=" * 70)

total_listings = df.shape[0]
avg_price = df['price'].mean()
median_price = df['price'].median()
avg_min_nights = df['minimum_nights'].mean()
avg_availability = df['availability_365'].mean()
avg_reviews = df['number_of_reviews'].mean()

print(f"Total Listings: {total_listings:,}")
print(f"Average Price: ${avg_price:.2f}")
print(f"Median Price: ${median_price:.2f}")
print(f"Avg Min Nights: {avg_min_nights:.1f}")
print(f"Avg Availability (days/year): {avg_availability:.1f}")
print(f"Avg Reviews per Listing: {avg_reviews:.1f}")

print("\n" + "=" * 70)
print("PRICE DISTRIBUTION BY NEIGHBOURHOOD GROUP")
print("=" * 70)

group_stats = df.groupby('neighbourhood_group').agg({
    'price': ['mean', 'median', 'count'],
    'availability_365': 'mean'
}).round(2)

print(group_stats)

print("\n" + "=" * 70)
print("PRICE BY ROOM TYPE")
print("=" * 70)

room_stats = df.groupby('room_type').agg({
    'price': ['mean', 'median', 'count']
}).round(2)

print(room_stats)

print("\n" + "=" * 70)
print("TOP 10 NEIGHBOURHOODS BY LISTING COUNT")
print("=" * 70)

neighbourhood_count = df['neighbourhood'].value_counts().head(10)
print(neighbourhood_count)

print("\n" + "=" * 70)
print("TOP 10 HOSTS BY LISTING COUNT")
print("=" * 70)

host_count = df.groupby('host_name')['calculated_host_listings_count'].sum().sort_values(ascending=False).head(10)
print(host_count)

df.to_csv(os.path.join(output_dir, 'summary_stats.csv'), index=False)

with open(os.path.join(output_dir, 'eda_report.txt'), 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("NYC AIRBNB EDA REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total Listings: {total_listings:,}\n")
    f.write(f"Average Price: ${avg_price:.2f}\n")
    f.write(f"Median Price: ${median_price:.2f}\n")
    f.write(f"Avg Min Nights: {avg_min_nights:.1f}\n")
    f.write(f"Avg Availability: {avg_availability:.1f}\n")
    f.write(f"Avg Reviews: {avg_reviews:.1f}\n")

print("\nEDA complete. Files saved to 3_outputs/")