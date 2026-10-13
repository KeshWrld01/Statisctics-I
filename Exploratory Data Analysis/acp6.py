# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import Dataset (Upload your student performance dataset CSV file when prompted)
from google.colab import files
uploaded = files.upload()

# Load the dataset (ensure the filename matches your uploaded file)
filename = list(uploaded.keys())[0]
data = pd.read_csv(filename)

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
# Task 3: Print All Statistical Measures of Numerical Features
# -------------------------------------------------------------
print("\n--- Statistical Measures ---")
print(data.describe())

# -------------------------------------------------------------
# Task 4: Create a Count Plot for all Categorical Variables
# -------------------------------------------------------------
categorical_cols = data.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x=col, palette='Set2')
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# -------------------------------------------------------------
# Task 5: Create a Pie Chart for 'Race/Ethnicity' and 'Parental Level of Education'
# -------------------------------------------------------------
# Note: Column names might vary slightly depending on your CSV (e.g., 'race/ethnicity', 'parental level of education')
race_col = [c for c in data.columns if 'race' in c.lower() or 'ethnicity' in c.lower()]
parent_col = [c for c in data.columns if 'parent' in c.lower()]

if race_col:
    plt.figure(figsize=(6, 6))
    data[race_col[0]].value_counts().plot(kind='pie', autopct='%.2f%%', colors=sns.color_palette('pastel'))
    plt.title('Pie Chart of Race/Ethnicity')
    plt.ylabel('')
    plt.show()

if parent_col:
    plt.figure(figsize=(6, 6))
    data[parent_col[0]].value_counts().plot(kind='pie', autopct='%.2f%%', colors=sns.color_palette('muted'))
    plt.title('Pie Chart of Parental Level of Education')
    plt.ylabel('')
    plt.show()

# -------------------------------------------------------------
# Task 6: How many males and females belong to different race groups?
# -------------------------------------------------------------
if race_col:
    gender_col = [c for c in data.columns if 'gender' in c.lower()][0]
    print("\n--- Gender count across Race Groups ---")
    print(pd.crosstab(data[gender_col], data[race_col[0]]))
    
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x=race_col[0], hue=gender_col, palette='coolwarm')
    plt.title('Gender Distribution across Race/Ethnicity Groups')
    plt.show()

# -------------------------------------------------------------
# Task 7: How many male and female students opted for different types of lunch?
# -------------------------------------------------------------
lunch_col = [c for c in data.columns if 'lunch' in c.lower()]
if lunch_col and gender_col:
    print("\n--- Gender count across Lunch Types ---")
    print(pd.crosstab(data[gender_col], data[lunch_col[0]]))
    
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x=lunch_col[0], hue=gender_col, palette='Set1')
    plt.title('Gender Distribution by Lunch Type')
    plt.show()

# -------------------------------------------------------------
# Task 8: Check Distribution and Skewness of Quantitative Features
# -------------------------------------------------------------
num_cols = data.select_dtypes(include=[np.number]).columns

for col in num_cols:
    skew_val = data[col].skew()
    print(f"Skewness of {col}: {skew_val:.2f}")
    
    plt.figure(figsize=(7, 4))
    sns.histplot(data[col], kde=True, color='teal')
    plt.title(f'Distribution of {col} (Skewness: {skew_val:.2f})')
    plt.show()

# -------------------------------------------------------------
# Task 9: Are features 'Gender' and 'Math Score' associated?
# -------------------------------------------------------------
math_col = [c for c in data.columns if 'math' in c.lower()][0]
if gender_col and math_col:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=gender_col, y=math_col, palette='Set3')
    plt.title('Association between Gender and Math Score')
    plt.show()

# -------------------------------------------------------------
# Task 10: Create a Heatmap to Check Correlation Between Features
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
numeric_data = data.select_dtypes(include=[np.number])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()