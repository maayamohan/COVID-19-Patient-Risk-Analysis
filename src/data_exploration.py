import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Covid Data.csv")

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== UNIQUE VALUES =====")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).head(15))