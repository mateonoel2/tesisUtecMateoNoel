from dist_fix import dist_fix
from sort_and_calc import sort_and_calc
from normalize import normalize
import pandas as pd
import pyarrow as pa
from dask.distributed import get_client, wait
import time

def process_data(ddf):
    while True:
        try:
            get_client()
            break
        except RuntimeError as e:
            print(f"Error: {e}. Waiting for new client...")
            time.sleep(1)


    first_time_received = ddf['time_received'].head(1).values[0]
    date = first_time_received[0:10]

    ddf = ddf.repartition(npartitions=88)

    ddf['vehicle_id'] = ddf['vehicle_id'].astype('int')

    # Filter out all phases that aren't LAYOVER_DURING and all rows with null
    ddf = ddf.loc[(ddf['inferred_phase'] == "IN_PROGRESS")].dropna()
    ddf = ddf.drop(columns='inferred_phase')

    # Apply the sort_and_calc() function to each group separately
    group = ddf.groupby('vehicle_id')
    ddf = group.apply(sort_and_calc, meta=pd.DataFrame(columns=['vehicle_id', 'trip', 'distance', 'date', 'exit_time', 'arrive_time']))
    ddf = ddf.reset_index(drop=True)

    group = ddf.groupby('trip')
    ddf = group.apply(dist_fix, meta=ddf._meta)
    ddf = ddf.reset_index(drop=True)

    group = ddf.groupby('vehicle_id')
    ddf = group.apply(normalize, meta=pd.DataFrame(columns=['total_distance','first_stop','exit_stop', 'target_stop', 'day_of_week', 'day_of_month', 'month','vehicle_id','distance', 'exit_time', 'arrive_time']))
    ddf = ddf.reset_index(drop=True)

    partition_schema = pa.schema([
        pa.field('vehicle_id', pa.int32()),
        pa.field('month', pa.int32()),
        pa.field('day_of_month', pa.int32()),
        pa.field('day_of_week', pa.int32()),
        pa.field('first_stop', pa.int64()),
        pa.field('exit_stop', pa.int64()),
        pa.field('target_stop', pa.int64()),
        pa.field('total_distance', pa.float64()),
        pa.field('distance', pa.float64()),
        pa.field('exit_time', pa.float64()),
        pa.field('arrive_time', pa.float64()),
    ])

    ddf = ddf.compute()
    ddf.to_parquet(f'../processed_datasets/{date}.parquet', engine='pyarrow', schema=partition_schema, compression='snappy')

    print(f"Finished processing {date} dataset.")

    return 0