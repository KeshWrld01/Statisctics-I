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
# Task 3: Find the Frequency of Categories for Feature 'Genre'
# -------------------------------------------------------------
print("\n--- Frequency of each Genre ---")
print(data['Genre'].value_counts())

# -------------------------------------------------------------
# Task 4: Find the Median Value of Feature 'Genre'
# -------------------------------------------------------------
# Get unique categories and convert 'Genre' to an ordered categorical type
genre_categories = data['Genre'].unique().tolist()
data['Genre_num'] = pd.Categorical(data['Genre'], categories=genre_categories, ordered=True)

# Find median category using categorical codes
median_index = np.median(data['Genre_num'].cat.codes)
median_genre = genre_categories[int(median_index)]

print("\n--- Median Value of Genre ---")
print("Median Genre:", median_genre)

# -------------------------------------------------------------
# Task 5: Create a Bar Chart and Pie Chart for Feature 'Genre'
# -------------------------------------------------------------
sns.set_style('whitegrid')

# Bar Chart for Genre
plt.figure(figsize=(8, 5))
sns.countplot(x='Genre', data=data, order=data['Genre'].value_counts().index, palette='winter')
plt.title('Bar Chart of Book Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

# Pie Chart for Genre
plt.figure(figsize=(6, 6))
data.groupby('Genre').size().plot(kind='pie', autopct='%.2f%%', colors=['skyblue', 'lightgreen'])
plt.title('Pie Chart of Book Genres')
plt.ylabel('') # Hide the y-label for better aesthetics
plt.show()

# -------------------------------------------------------------
# Task 6: Create Numerical Data Frame and Check Spread via Boxplot
# -------------------------------------------------------------
# Select only numerical features
num_data = data.select_dtypes(include=[np.number]).drop(columns=['Genre_num'], errors='ignore')

print("\n--- Numerical Features ---")
print(num_data.columns.tolist())

# Create boxplots for each numerical feature to check for outliers and spread
for col in num_data.columns:
    plt.figure(figsize=(8, 4))
    plt.boxplot(num_data[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
    plt.show()

# -------------------------------------------------------------
# Task 7: Transform Data Using Standardization
# -------------------------------------------------------------
# Since boxplots often reveal outliers in numerical columns like 'Price' or 'Reviews',
# StandardScaler is preferred for robust transformation.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_data)

# Convert back to a DataFrame for viewing
scaled_df = pd.DataFrame(scaled_data, columns=num_data.columns)
print("\n--- First 5 rows of Scaled Numerical Data ---")
print(scaled_df.head())