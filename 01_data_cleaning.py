import pandas as pd
import numpy as np

df = pd.read_csv('1_data/retail_sales_dataset.csv')

df.head()
df.info()
df.describe()

df.isnull().sum()

df.duplicated().sum()

df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.day_name()

df['Total Amount'].describe()

df['Age'].describe()

df['Price per Unit'].describe()

df['Quantity'].describe()

df.to_csv('3_outputs/cleaned_retail_sales.csv', index=False)

df_cleaned = pd.read_csv('3_outputs/cleaned_retail_sales.csv')
df_cleaned.head()