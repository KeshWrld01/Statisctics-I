# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import Dataset (Upload your 'bestsellers.csv' or dataset file when prompted)
from google.colab import files
uploaded = files.upload()

# Load the dataset (make sure the filename matches your uploaded file)
data = pd.read_csv('bestsellers.csv')

# Display the first few rows
data.head()

# -------------------------------------------------------------
# Task 1: Check if any null values are present in the dataset
# -------------------------------------------------------------
print("--- Missing Values Before Handling ---")
print(data.isnull().sum())

# -------------------------------------------------------------
# Task 2: Handle missing values if any
# -------------------------------------------------------------
# Filling missing numerical values with the median
for col in data.select_dtypes(include=[np.number]).columns:
    data[col] = data[col].fillna(data[col].median())

# Filling missing categorical values with the mode
for col in data.select_dtypes(include=['object', 'category']).columns:
    if not data[col].mode().empty:
        data[col] = data[col].fillna(data[col].mode()[0])

print("\n--- Missing Values After Handling ---")
print(data.isnull().sum())

# -------------------------------------------------------------
# Task 3: Find the value that divides 10% of data for feature Price
# -------------------------------------------------------------
price_10_percent = np.quantile(data['Price'], 0.10)
print("\n--- Quantile Calculation ---")
print("Value that splits 10% of the Price data:", price_10_percent)

# -------------------------------------------------------------
# Task 4: Find 1st, 2nd, 3rd Quartiles and IQR of feature 'Price'
# -------------------------------------------------------------
price_q1 = np.quantile(data['Price'], 0.25)
price_q2 = np.quantile(data['Price'], 0.50)
price_q3 = np.quantile(data['Price'], 0.75)
iqr_price = price_q3 - price_q1

print("\n--- Price Quartiles and IQR ---")
print("Q1 (25%):", price_q1)
print("Q2 (50% / Median):", price_q2)
print("Q3 (75%):", price_q3)
print("Interquartile Range (IQR):", iqr_price)

# -------------------------------------------------------------
# Task 5: Find 1st, 2nd, 3rd Quartiles and IQR of feature 'User Rating'
# -------------------------------------------------------------
rating_q1 = np.quantile(data['User Rating'], 0.25)
rating_q2 = np.quantile(data['User Rating'], 0.50)
rating_q3 = np.quantile(data['User Rating'], 0.75)
iqr_rating = rating_q3 - rating_q1

print("\n--- User Rating Quartiles and IQR ---")
print("Q1 (25%):", rating_q1)
print("Q2 (50% / Median):", rating_q2)
print("Q3 (75%):", rating_q3)
print("Interquartile Range (IQR):", iqr_rating)

# -------------------------------------------------------------
# Task 6: Create boxplots for features 'Price' and 'User Rating'
# -------------------------------------------------------------
# Boxplot for Price
plt.figure(figsize=(8, 5))
plt.boxplot(data['Price'])
plt.title('Boxplot of Book Price')
plt.ylabel('Price ($)')
plt.show()

# Boxplot for User Rating
plt.figure(figsize=(8, 5))
plt.boxplot(data['User Rating'])
plt.title('Boxplot of User Rating')
plt.ylabel('User Rating')
plt.show()