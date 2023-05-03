import dask.dataframe as dd
import os
from calc_data import calc_data
from normalize import normalize
import pandas as pd
from dask.distributed import Client
from pyarrow import schema
import pyarrow as pa
import logging

# Define a function to sort the partition by 'time_received'
def sort_and_calc(df):
    sorted_df = df.sort_values('time_received')
    return calc_data(sorted_df)

def dist_fix(df):
    df.loc[:,'distance'] = df['distance'].round(decimals=-1)
    mode = df['distance'].mode()[0]
    df = df[(df['distance'] > mode - 20) & (df['distance'] < mode + 20)]
    df.loc[:, 'distance'] = mode
    return df

if __name__ == '__main__':
    
    client = Client()

    folder = "../datasets"
    datasets = os.listdir(folder)

    # Load your Dask DataFrame
    dataset_names = []
    for dataset in datasets: 
        dataset_names.append(os.path.join(folder, dataset))

    try:
        ddf = dd.read_csv(dataset_names, sep="\t", assume_missing=True, usecols=['time_received', 'vehicle_id', 'distance_along_trip', 'inferred_phase', 'next_scheduled_stop_distance', 'next_scheduled_stop_id'])

        # Filter out all phases that aren't LAYOVER_DURING and all rows with null
        ddf = ddf.loc[(ddf['inferred_phase'] != "LAYOVER_DURING")].dropna()

        #Test filter line
        ddf = ddf.loc[(ddf['vehicle_id'] == 469.0) | (ddf['vehicle_id'] == 195.0)]

        # Apply the sort_and_calc() function to each group separately
        group = ddf.groupby('vehicle_id')
        ddf = group.apply(sort_and_calc, meta=pd.DataFrame(columns=['trip', 'distance', 'date', 'exit_time', 'arrive_time']))
        ddf = ddf.reset_index(drop=True)

        # Apply the dist_fix() function to each group separately
        group = ddf.groupby('trip')
        ddf = group.apply(dist_fix, meta=ddf._meta)
        ddf = ddf.reset_index(drop=True)

        ddf = ddf.map_partitions(normalize, meta=pd.DataFrame(columns=['exit_stop', 'target_stop', 'day_of_week', 'day_of_month', 'month','distance', 'exit_time', 'arrive_time']))
        
        partition_schema = pa.schema([
            pa.field('exit_stop', pa.int64()),
            pa.field('target_stop', pa.int64()),
            pa.field('day_of_week', pa.int32()),
            pa.field('day_of_month', pa.int32()),
            pa.field('month', pa.int32()),
            pa.field('distance', pa.float64()),
            pa.field('exit_time', pa.float64()),
            pa.field('arrive_time', pa.float64()),
            pa.field('__null_dask_index__', pa.int64())
        ])
        
        # save the Pandas dataframe in Parquet format
        ddf.to_parquet('../processed_datasets/test.parquet', engine='pyarrow', schema=partition_schema)
    except dd.core.ShuffleError as e:
        logging.error(f"An error occurred: {e}")
        with open('error.log', 'a') as f:
            f.write
