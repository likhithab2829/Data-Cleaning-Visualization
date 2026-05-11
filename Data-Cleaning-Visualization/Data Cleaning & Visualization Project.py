# ============================================
# DATA CLEANING & VISUALIZATION PROJECT
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# STEP 1: LOAD DATASET
# ============================================

# Load dataset
# Replace 'sales_data.csv' with your dataset file
df = pd.read_csv("sales_data.csv")

# Display first 5 rows
print("FIRST 5 ROWS OF DATASET")
print(df.head())

# Dataset Information
print("\nDATASET INFO")
print(df.info())

# ============================================
# STEP 2: CHECK MISSING VALUES
# ============================================

print("\nMISSING VALUES")
print(df.isnull().sum())

# Fill missing numeric values with mean
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill missing categorical values with mode
categorical_columns = df.select_dtypes(include='object').columns

for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# ============================================
# STEP 3: REMOVE DUPLICATES
# ============================================

print("\nNUMBER OF DUPLICATES:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("DUPLICATES REMOVED")

# ============================================
# STEP 4: HANDLE OUTLIERS
# ============================================

# Using IQR Method
for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_limit) &
            (df[col] <= upper_limit)]

print("\nOUTLIERS REMOVED")

# ============================================
# STEP 5: DATA SUMMARY
# ============================================

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# ============================================
# STEP 6: DATA VISUALIZATION
# ============================================

# Set style
sns.set_style("whitegrid")

# --------------------------------------------
# HISTOGRAM
# --------------------------------------------

for col in numeric_columns:

    plt.figure(figsize=(8,5))

    sns.histplot(df[col], kde=True)

    plt.title(f"Distribution of {col}")

    plt.xlabel(col)

    plt.ylabel("Frequency")

    plt.show()

# --------------------------------------------
# BOXPLOT
# --------------------------------------------

for col in numeric_columns:

    plt.figure(figsize=(8,5))

    sns.boxplot(x=df[col])

    plt.title(f"Boxplot of {col}")

    plt.show()

# --------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------

plt.figure(figsize=(10,8))

correlation = df.corr(numeric_only=True)

sns.heatmap(correlation,
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

# --------------------------------------------
# BAR CHART FOR CATEGORICAL DATA
# --------------------------------------------

for col in categorical_columns:

    plt.figure(figsize=(8,5))

    df[col].value_counts().plot(kind='bar')

    plt.title(f"Count of {col}")

    plt.xlabel(col)

    plt.ylabel("Count")

    plt.show()

# ============================================
# STEP 7: SAVE CLEANED DATA
# ============================================

df.to_csv("cleaned_data.csv", index=False)

print("\nCLEANED DATA SAVED SUCCESSFULLY")

# ============================================
# STEP 8: FINAL INSIGHTS
# ============================================

print("\nPROJECT COMPLETED SUCCESSFULLY")

print("""
Key Tasks Performed:
1. Loaded dataset
2. Handled missing values
3. Removed duplicates
4. Removed outliers
5. Generated statistics
6. Created visualizations
7. Saved cleaned dataset
""")