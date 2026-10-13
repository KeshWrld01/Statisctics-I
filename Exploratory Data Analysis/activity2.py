# ============================================================
# Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Import Dataset
# ============================================================

from google.colab import files

uploaded = files.upload()

# Automatically use the uploaded file
filename = list(uploaded.keys())[0]

# Import dataset
data = pd.read_csv(filename)

data.head(5)


# ============================================================
# Check Null Values
# ============================================================

data.isnull().sum()


# ============================================================
# Statistical Information of Features
# ============================================================

data.describe()


# ============================================================
# Create Boxplot of all the numerical features
# and check if outliers are present
# ============================================================

labels = [
    'Id',
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]

for label in labels:
    print('Distribution of', label)

    sns.boxplot(x=data[label])

    plt.show()


# ============================================================
# Conclusion
# ============================================================
# SepalWidthCm has outliers


# ============================================================
# Check Correlation Between All Numerical Features
# ============================================================

# Select only numerical columns
numeric_data = data.select_dtypes(include=['number'])

plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()


# ============================================================
# Check if any feature is skewed or not
# ============================================================

labels = [
    'Id',
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]

for label in labels:

    print('Distribution of', label)

    sns.histplot(
        data[label],
        kde=True
    )

    plt.show()


# ============================================================
# Check Skewness of Each Feature
# ============================================================

labels = [
    'Id',
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]

for label in labels:

    print('Skewness of', label)

    print(data[label].skew())