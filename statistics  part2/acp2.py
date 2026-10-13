# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import Dataset (Upload your 'bestsellers.csv' or dataset file when prompted)
from google.colab import files
uploaded = files.upload()

# Load the dataset (ensure the filename matches your uploaded file)
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
# Task 3: Find Variance and Standard Deviation of 'User Rating'
# -------------------------------------------------------------
var_rating = np.var(data['User Rating'])
std_rating = np.std(data['User Rating'])

print("\n--- User Rating Distribution ---")
print("Variance of User Rating:", var_rating)
print("Standard Deviation of User Rating:", std_rating)

# -------------------------------------------------------------
# Task 4: Find Variance and Standard Deviation of 'Price'
# -------------------------------------------------------------
var_price = np.var(data['Price'])
std_price = np.std(data['Price'])

print("\n--- Price Distribution ---")
print("Variance of Price:", var_price)
print("Standard Deviation of Price:", std_price)

# -------------------------------------------------------------
# Task 5: Check distribution of 'User Rating' using a customized histogram
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
bins_rating = np.arange(data['User Rating'].min(), data['User Rating'].max() + 0.2, 0.2)
plt.hist(data['User Rating'], bins=bins_rating, color='skyblue', edgecolor='black')
plt.title("Distribution of User Ratings")
plt.xlabel("User Rating")
plt.ylabel("Frequency")
plt.xticks(bins_rating)
plt.show()

# -------------------------------------------------------------
# Task 6: Check distribution of 'Price' using a customized histogram
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
bins_price = np.arange(data['Price'].min(), data['Price'].max() + 5, 5)
plt.hist(data['Price'], bins=bins_price, color='lightgreen', edgecolor='black')
plt.title("Distribution of Book Prices")
plt.xlabel("Price ($)")
plt.ylabel("Frequency")
plt.show()