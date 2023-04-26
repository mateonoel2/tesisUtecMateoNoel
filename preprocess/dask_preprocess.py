import dask.dataframe as dd
import os
from calc_data import calc_data
import pandas as pd
from dask.distributed import Client

# Define a function to sort the partition by 'time_received'
def sort_and_calc(df):
    sorted_df = df.sort_values('time_received')
    return calc_data(sorted_df)

if __name__ == '__main__':
    client = Client() 

    folder = "../datasets"
    datasets = os.listdir(folder)

    # Load your Dask DataFrame
    dataset_name = os.path.join(folder, datasets[0])
    ddf = dd.read_csv(dataset_name, sep="\t", assume_missing=True)

    # Filter out and drop missing values
    ddf = ddf.loc[(ddf['inferred_phase'] != "LAYOVER_DURING") & 
                (ddf['vehicle_id'].notnull()) & 
                (ddf['time_received'].notnull()) & 
                (ddf['next_scheduled_stop_distance'].notnull()) & 
                (ddf['distance_along_trip'].notnull())]


    #Test filter line
    #ddf = ddf.loc[(ddf['vehicle_id'] == 469.0) | (ddf['vehicle_id'] == 195.0)]

    # Get the number of unique vehicle_id values
    n_partitions = ddf['vehicle_id'].nunique().compute(num_workers=8)

    # Set the index to the 'vehicle_id' column
    ddf = ddf.set_index('vehicle_id')

    # Repartition the data based on the index
    ddf = ddf.repartition(npartitions=n_partitions)

    # Apply the combined function to each partition and compute the result
    data = ddf.map_partitions(sort_and_calc, meta=ddf._meta).compute(num_workers=8)

    # create an empty dataframe
    columns = ['exit_stop', 'target_stop', 'distance', 'speed', 'date', 'weekday', 'exit_time', 'arrive_time']
    dfs = []

    # iterate over each row in the series
    for row in data:
        dfs.append(pd.DataFrame(row, columns=columns))

    df = pd.concat(dfs, ignore_index=True)
    
    date = os.path.basename(dataset_name)[14:24]
    df.to_csv(f'../preprocessed_datasets/{date}.csv', index=False)