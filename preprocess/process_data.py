from dist_fix import dist_fix
from sort_and_calc import sort_and_calc
from normalize import normalize
import pandas as pd
import pyarrow as pa
from dask.distributed import get_client
import time
import sys

def process_data(ddf):

    while True:
        try:
            get_client()
            break
        except RuntimeError as e:
            print(f"Error: {e}. Waiting for new client...")
            time.sleep(1)

    # Get the first value of the time_received column
    first_time_received = ddf['time_received'].head(1).values[0]
    date = first_time_received[0:10]

    # Filter out all phases that aren't LAYOVER_DURING and all rows with null
    ddf = ddf.loc[(ddf['inferred_phase'] == "IN_PROGRESS")].dropna()
    ddf = ddf.loc[(ddf['vehicle_id'] == 202.0)] 
    ddf = ddf.drop(columns='inferred_phase')

    # Apply the sort_and_calc() function to each group separately
    group = ddf.groupby('vehicle_id')
    ddf = group.apply(sort_and_calc, meta=pd.DataFrame(columns=['first_stop','trip', 'distance','total_distance', 'date', 'exit_time', 'arrive_time']))
    ddf = ddf.reset_index(drop=True)

    # Apply the dist_fix() function to each group separately
    group = ddf.groupby('trip')
    ddf = group.apply(dist_fix, meta=ddf._meta)
    ddf = ddf.reset_index(drop=True)

    ddf = ddf.map_partitions(normalize, meta=pd.DataFrame(columns=['exit_stop', 'target_stop', 'day_of_week', 'day_of_month', 'month','first_stop', 'distance','total_distance','exit_time', 'arrive_time']))

    partition_schema = pa.schema([
        pa.field('exit_stop', pa.int64()),
        pa.field('target_stop', pa.int64()),
        pa.field('day_of_week', pa.int32()),
        pa.field('day_of_month', pa.int32()),
        pa.field('month', pa.int32()),
        pa.field('first_stop', pa.int64()),
        pa.field('distance', pa.float64()),
        pa.field('total_distance', pa.float64()),
        pa.field('exit_time', pa.float64()),
        pa.field('arrive_time', pa.float64()),
        pa.field('__null_dask_index__', pa.int64())
    ])
 
    # Write processed data to file
    ddf.to_parquet(f'../processed_datasets/{date}.parquet', engine='pyarrow', schema=partition_schema)

    print(f"Finished processing {date} dataset.")

    return None