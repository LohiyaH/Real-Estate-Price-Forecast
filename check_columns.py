import pandas as pd

# Load the data
df = pd.read_csv("train.csv")

# Print column names
print("Column names:", df.columns.tolist())
