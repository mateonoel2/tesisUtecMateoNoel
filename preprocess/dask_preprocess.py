import dask.dataframe as dd
import os
from calc_data import calc_data
import pandas as pd

folder = "../datasets"
datasets = os.listdir(folder)

# Load your Dask DataFrame
ddf = dd.read_csv(os.path.join(folder, datasets[0]), sep="\t", assume_missing=True)

# Filter out and drop missing values
ddf = ddf.loc[~(ddf['inferred_phase'] == 'LAYOVER_DURING')]
ddf = ddf.dropna(subset=['vehicle_id', 'time_received', 'next_scheduled_stop_distance', 'distance_along_trip'])
ddf = ddf.loc[(ddf['vehicle_id'] == 469.0) | (ddf['vehicle_id'] == 195.0)]

# Get the number of unique vehicle_id values
n_partitions = ddf['vehicle_id'].nunique().compute()

# Set the index to the 'vehicle_id' column
ddf = ddf.set_index('vehicle_id')

# Repartition the data based on the index
ddf = ddf.repartition(npartitions=n_partitions)

# Define a function to sort the partition by 'time_received'
def sort_partition(df):
    return df.sort_values('time_received')

ddf = ddf.map_partitions(sort_partition)

# Apply the function to each partition
data = ddf.map_partitions(calc_data, meta=ddf._meta).compute()

# create an empty dataframe
columns = ['current_stop', 'next_stop', 'distance', 'start_time', 'end_time', 'speed']
df = pd.DataFrame(columns=columns)

# iterate over each row in the series
for row in data:
    # create a temporary dataframe for the current row
    temp_df = pd.DataFrame(row, columns=columns)
    
    # append the temporary dataframe to the main dataframe
    df = df.append(temp_df, ignore_index=True)

# write the dataframe to a csv file
df.to_csv('results/result.csv', index=False)