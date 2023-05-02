import pandas as pd

# Load the parquet file into a pandas dataframe
df = pd.read_parquet('processed_datasets/test.parquet')

# Save the dataframe as a CSV file
df.to_csv('processed_datasets/test2.csv', index=False)
