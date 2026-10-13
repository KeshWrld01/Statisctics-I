# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import Dataset (Upload your 'bestsellers.csv' file when prompted)
from google.colab import files
uploaded = files.upload()

# Load the dataset (ensure the filename matches your uploaded file)
data = pd.read_csv('bestsellers.csv')

# Display the first few rows
data.head()

# -------------------------------------------------------------
# Task 1 & 2: Check and Handle Missing Values
# -------------------------------------------------------------
print("--- Missing Values Before Handling ---")
print(data.isnull().sum())

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
# Task 3: Check if there is a requirement of combining categories
# -------------------------------------------------------------
print("\n--- Genre Value Counts ---")
print(data['Genre'].value_counts())
# Conclusion: There are only 2 distinct categories (Fiction, Non Fiction) in 'Genre', 
# so combining or collapsing categories is not required.

# -------------------------------------------------------------
# Task 4: Bin the values of feature 'User Rating' into Categories ('Good', 'Excellent')
# -------------------------------------------------------------
# Find min and max for User Rating to establish sensible bins
min_rating = data['User Rating'].min()
max_rating = data['User Rating'].max()

# Define bin boundaries and labels
rating_bins = [min_rating - 0.1, 4.5, max_rating]
rating_labels = ['Good', 'Excellent']

data['binned_user_rating'] = pd.cut(data['User Rating'], bins=rating_bins, labels=rating_labels)
print("\n--- Binned User Rating Head ---")
print(data[['User Rating', 'binned_user_rating']].head())

# -------------------------------------------------------------
# Task 5: Check Skewness of Numerical Features
# -------------------------------------------------------------
print("\n--- Skewness of Numerical Features ---")
num_cols = data.select_dtypes(include=[np.number]).columns

for col in num_cols:
    skew_val = data[col].skew()
    print(f"Skewness of {col}: {skew_val}")
    
    # Optional: Plot distribution to visualize skewness
    plt.figure(figsize=(6, 3))
    sns.histplot(data[col], kde=True, color='purple')
    plt.title(f'Distribution of {col} (Skewness: {skew_val:.2f})')
    plt.show()

# -------------------------------------------------------------
# Task 6: Check Association between User Rating and Genre
# -------------------------------------------------------------
# Using a boxplot with the categorical variable on x-axis and quantitative variable on y-axis
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x='Genre', y='User Rating', palette='Set2')
plt.title('Association between Genre and User Rating')
plt.xlabel('Genre')
plt.ylabel('User Rating')
plt.show()

# -------------------------------------------------------------
# Task 7: Check Association between User Rating and Price
# -------------------------------------------------------------
# Using a scatter plot between two quantitative variables
plt.figure(figsize=(8, 5))
plt.scatter(data['Price'], data['User Rating'], alpha=0.5, color='teal')
plt.title('Association between Price and User Rating')
plt.xlabel('Price ($)')
plt.ylabel('User Rating')
plt.show()