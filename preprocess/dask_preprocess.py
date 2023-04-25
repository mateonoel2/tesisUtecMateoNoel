import dask.dataframe as dd
import os
from calc_data import calc_data
import pandas as pd

# Define a function to sort the partition by 'time_received'
def sort_partition(df):
    return df.sort_values('time_received')

if __name__ == '__main__':
    folder = "../datasets"
    datasets = os.listdir(folder)

    # Load your Dask DataFrame
    dataset_name = os.path.join(folder, datasets[0])
    ddf = dd.read_csv(dataset_name, sep="\t", assume_missing=True)

    # Filter out and drop missing values
    ddf = ddf.query('inferred_phase != "LAYOVER_DURING"').dropna(subset=['vehicle_id', 'time_received', 'next_scheduled_stop_distance', 'distance_along_trip'])

    #Filter line
    ddf = ddf.loc[(ddf['vehicle_id'] == 469.0) | (ddf['vehicle_id'] == 195.0)]

    # Get the number of unique vehicle_id values
    n_partitions = ddf['vehicle_id'].nunique().compute()

    # Set the index to the 'vehicle_id' column
    ddf = ddf.set_index('vehicle_id')

    # Repartition the data based on the index
    ddf = ddf.repartition(npartitions=n_partitions)
    ddf = ddf.map_partitions(sort_partition)

    # Apply the function to each partition
    data = ddf.map_partitions(calc_data, meta=ddf._meta).compute()

    # create an empty dataframe
    columns = ['current_stop', 'next_stop', 'distance', 'start_time', 'end_time', 'speed']
    dfs = []

    # iterate over each row in the series
    for row in data:
        dfs.append(pd.DataFrame(row, columns=columns))

    df = pd.concat(dfs, ignore_index=True)

    date = os.path.basename(dataset_name)[14:24]
    df.to_csv(f'../processed_models/{date}.csv', index=False)