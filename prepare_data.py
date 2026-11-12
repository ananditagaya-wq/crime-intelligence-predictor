
import pandas as pd

# Load original dataset
df = pd.read_csv("data/crime_dataset.csv")

TARGET = "total_ipc_crimes"
KEYS = ["state_ut", "district", "year"]

# 1. Aggregate duplicate district-year records
numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

df = (
    df.groupby(KEYS, as_index=False)[numeric_columns]
    .sum()
)

df = df.sort_values(KEYS).reset_index(drop=True)

print("After aggregation:", df.shape)
print("Remaining duplicate keys:",
      df.duplicated(KEYS).sum())

# 2. Create historical features
# Use only actual preceding calendar years.
previous = df[KEYS + [TARGET]].copy()

previous_1 = previous.copy()
previous_1["year"] += 1
previous_1 = previous_1.rename(
    columns={TARGET: "ipc_lag_1"}
)

previous_2 = previous.copy()
previous_2["year"] += 2
previous_2 = previous_2.rename(
    columns={TARGET: "ipc_lag_2"}
)

previous_2["historical_avg_2"] = (
    previous_2["ipc_lag_2"]
)

# 3. Join historical features
data = df.merge(
    previous_1,
    on=KEYS,
    how="left"
)

data = data.merge(
    previous_2,
    on=KEYS,
    how="left"
)

# Average of the previous two years
data["historical_avg_2"] = (
    data["ipc_lag_1"] + data["ipc_lag_2"]
) / 2

# 4. Remove records without both historical years
data = data.dropna(
    subset=["ipc_lag_1", "ipc_lag_2"]
).copy()

# 5. Keep only model features and target
model_data = data[
    [
        "state_ut",
        "district",
        "year",
        "ipc_lag_1",
        "ipc_lag_2",
        "historical_avg_2",
        TARGET
    ]
].copy()

# 6. Chronological split
train = model_data[model_data["year"] <= 2009]
test = model_data[model_data["year"] >= 2010]

# 7. Save prepared datasets
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
model_data.to_csv(
    "data/model_dataset.csv", index=False
)

print("\n========== PREPARATION RESULTS ==========")
print("Total supervised records:", len(model_data))
print("Training records:", len(train))
print("Testing records:", len(test))

print("\nTraining year range:",
      train["year"].min(), "-", train["year"].max())

print("Testing year range:",
      test["year"].min(), "-", test["year"].max())

print("\nTraining features:")
print(train.columns.tolist())

print("\nSaved train.csv, test.csv and model_dataset.csv")
