
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/ai4i2020.csv")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

# -----------------------------
# 2. First 5 Rows
# -----------------------------
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# -----------------------------
# 3. Column Names
# -----------------------------
print("\n========== COLUMNS ==========")
print(df.columns.tolist())

# -----------------------------
# 4. Data Types
# -----------------------------
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# -----------------------------
# 5. Missing Values
# -----------------------------
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# -----------------------------
# 6. Duplicate Rows
# -----------------------------
print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

# -----------------------------
# 7. Machine Failure Distribution
# -----------------------------
print("\n========== MACHINE FAILURE ==========")
print(df["Machine failure"].value_counts())

# -----------------------------
# 8. Statistical Summary
# -----------------------------
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# -----------------------------
# 9. Failure Distribution Plot
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Machine failure")
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Machines")
plt.show()

# -----------------------------
# 10. Numerical Feature Distributions
# -----------------------------
numeric_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

for column in numeric_columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# -----------------------------
# 11. Correlation Matrix
# -----------------------------
plt.figure(figsize=(10, 7))

correlation = df[numeric_columns + ["Machine failure"]].corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Matrix")
plt.show()

print("\n========== EDA COMPLETED ==========")