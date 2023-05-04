import dask.dataframe as dd
import os
from normalize import normalize
from dist_fix import dist_fix
from sort_and_calc import sort_and_calc
import pandas as pd
import pyarrow as pa
from dask.distributed import Client, get_client
import dask.bag as db
import time
import dask.config

dask.config.set({'distributed.comm.timeouts.connect': '120m'})

if __name__ == '__main__':    

    client = Client(timeout="3h")

    folder = "../datasets"
    datasets = os.listdir(folder)
    dataset_names = []

    for dataset in datasets:
        dataset_names.append(os.path.join(folder, dataset))
    
    # Create a Dask bag of Dask dataframes
    dataframes_bag = db.from_sequence(dataset_names).map(lambda fn: dd.read_csv(fn, sep="\t", assume_missing=True, usecols=['time_received', 'vehicle_id', 'distance_along_trip', 'inferred_phase', 'next_scheduled_stop_distance', 'next_scheduled_stop_id']))
    
    def process_data(ddf):

        while True:
            try:
                client = get_client()
                break
            except RuntimeError as e:
                print(f"Error: {e}. Waiting for new client...")
                time.sleep(1)
        
        # Get the first value of the time_received column
        first_time_received = ddf['time_received'].head(1).values[0]
        date = first_time_received[0:10]

        # Filter out all phases that aren't LAYOVER_DURING and all rows with null
        ddf = ddf.loc[(ddf['inferred_phase'] != "LAYOVER_DURING")].dropna()

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
     
        # Write processed data to file
        ddf.to_parquet(f'../processed_datasets/{date}.parquet', engine='pyarrow', schema=partition_schema)

        print(f"Finished processing {date} dataset.")

        return None

    dataframes_bag.map(process_data).compute()

    client.close()
