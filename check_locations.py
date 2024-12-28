import pandas as pd

# Load the data
df = pd.read_csv("train.csv")

# Get unique locations
locations = sorted(df['ADDRESS'].unique())
print("Available locations in the dataset:")
for loc in locations:
    print(f'<option value="{loc}">{loc}</option>')
