
import pandas as pd

# Load dataset
df = pd.read_csv("data/crime_dataset.csv")

print("\n========== DATASET OVERVIEW ==========")

print("\nShape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\n========== TARGET COLUMN CHECK ==========")

target = "TOTAL IPC CRIMES"

if target in df.columns:
    print(df[target].describe())
else:
    print("Target not found. Check actual column name.")
