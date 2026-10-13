# Import Libraries
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import Dataset (Upload your 'Bestseller Books' CSV file when prompted)
from google.colab import files
uploaded = files.upload()

# Load the dataset (replace 'bestsellers.csv' with your exact uploaded file name if different)
data = pd.read_csv('bestsellers.csv')

# Display the first few rows to inspect the dataset
data.head()

# -------------------------------------------------------------
# Task 1: Check quantitative and categorical features
# -------------------------------------------------------------
print("--- Data Types ---")
print(data.dtypes)

# -------------------------------------------------------------
# Task 2 & 3: Check and handle missing (null) values if any
# -------------------------------------------------------------
print("\n--- Missing Values Before Handling ---")
print(data.isnull().sum())

# Fill missing numerical values with the median or drop them if needed. 
# For example, filling missing values in quantitative columns with their median:
for col in data.select_dtypes(include=[np.number]).columns:
    data[col] = data[col].fillna(data[col].median())

# For categorical columns, you can fill missing values with the mode:
for col in data.select_dtypes(include=['object', 'category']).columns:
    if not data[col].mode().empty:
        data[col] = data[col].fillna(data[col].mode()[0])

print("\n--- Missing Values After Handling ---")
print(data.isnull().sum())

# -------------------------------------------------------------
# Task 4: Find Mean, Median, and Mode for 'User Rating'
# -------------------------------------------------------------
mean_rating = np.mean(data['User Rating'])
median_rating = np.median(data['User Rating'])
mode_rating = stats.mode(data['User Rating'])

print("\n--- User Rating Statistics ---")
print("Mean User Rating:", mean_rating)
print("Median User Rating:", median_rating)
print("Mode User Rating:", mode_rating)

# -------------------------------------------------------------
# Task 5: Find Mean, Median, and Mode for 'Price'
# -------------------------------------------------------------
mean_price = np.mean(data['Price'])
median_price = np.median(data['Price'])
mode_price = stats.mode(data['Price'])

print("\n--- Price Statistics ---")
print("Mean Price:", mean_price)
print("Median Price:", median_price)
print("Mode Price:", mode_price)

# -------------------------------------------------------------
# Task 6: Find Mean, Median, and Mode for 'Reviews'
# -------------------------------------------------------------
mean_reviews = np.mean(data['Reviews'])
median_reviews = np.median(data['Reviews'])
mode_reviews = stats.mode(data['Reviews'])

print("\n--- Reviews Statistics ---")
print("Mean Reviews:", mean_reviews)
print("Median Reviews:", median_reviews)
print("Mode Reviews:", mode_reviews)