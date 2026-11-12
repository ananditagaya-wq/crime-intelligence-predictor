
import pandas as pd

df = pd.read_csv("data/crime_dataset.csv")

print("\n========== CRIME DATA AUDIT ==========")

# 1. Target column
target = "total_ipc_crimes"

print("\nTarget statistics:")
print(df[target].describe())

# 2. Check duplicate district-year combinations
duplicates = df.duplicated(
    subset=["state_ut", "district", "year"],
    keep=False
)

print("\nDuplicate district-year records:", duplicates.sum())

if duplicates.any():
    print(df.loc[duplicates, [
        "state_ut", "district", "year", target
    ]].sort_values(["state_ut", "district", "year"]))

# 3. Check whether total equals sum of crime components
crime_columns = [
    col for col in df.select_dtypes(include="number").columns
    if col not in ["year", target]
]

df["component_sum"] = df[crime_columns].sum(axis=1)

df["difference"] = (
    df[target] - df["component_sum"]
)

print("\nComponent sum vs total:")
print("Exact matches:", (df["difference"] == 0).sum())
print("Non-matches:", (df["difference"] != 0).sum())

# 4. Check historical records per district
history = (
    df.groupby(["state_ut", "district"])["year"]
    .nunique()
)

print("\nDistrict history distribution:")
print(history.describe())

print("\nDistricts with fewer than 2 years:",
      (history < 2).sum())

print("\n========== AUDIT COMPLETED ==========")
