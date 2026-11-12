
import pandas as pd

df = pd.read_csv("data/crime_dataset.csv")

print("\n========== DUPLICATE INVESTIGATION ==========")

# Find repeated state-district-year combinations
duplicates = df[
    df.duplicated(
        subset=["state_ut", "district", "year"],
        keep=False
    )
]

# Display every column for the duplicate records
print("\nComplete duplicate-key records:")
print(duplicates.T.to_string())

# Check whether the duplicate records differ
if not duplicates.empty:
    print("\nDifferences between duplicate records:")
    print(duplicates.iloc[0].compare(duplicates.iloc[1]))

# Check year coverage
print("\nRecords per year:")
print(df.groupby("year").size())

# Check district history
history = (
    df.groupby(["state_ut", "district"])["year"]
    .nunique()
)

print("\nDistricts by number of available years:")
print(history.value_counts().sort_index())

print("\n========== INVESTIGATION COMPLETE ==========")
