from dist_fix import dist_fix
from sort_and_calc import sort_and_calc
from normalize import normalize
import pandas as pd
import pyarrow as pa
import dask.dataframe as dd
from dask.distributed import get_client
import time

def total_distances(ddf, routes):
    ddf.insert(0, 'first_stop', None)
    ddf.insert(0, 'total_distance', 0)
    
  
    while ddf['first_stop'].isna().any():
        for route_name, stops in routes.items():
            mask = ddf['exit_stop'].isin(stops) & ddf['first_stop'].isna()
            mask2 = ddf['exit_stop'].isin(stops)
            total_distance = ddf.loc[mask2, 'total_distance'].max()

            new_stops = ddf.loc[mask, 'target_stop'].to_list()   
            ddf.loc[mask, 'total_distance'] = ddf.loc[mask, 'distance'] + total_distance
            routes[route_name] = routes[route_name] | set(new_stops)
            ddf.loc[mask,'first_stop'] = route_name
    
    return ddf

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
    ddf = ddf.drop(columns='inferred_phase')

    #use only two vehicles
    #ddf = ddf.loc[(ddf['vehicle_id'] == 469.0) | (ddf['vehicle_id'] == 195.0)]

    # Apply the sort_and_calc() function to each group separately
    group_v = ddf.groupby('vehicle_id')
    ddf = group_v.apply(sort_and_calc, meta=pd.DataFrame(columns=['vehicle_id', 'trip', 'distance', 'date', 'exit_time', 'arrive_time']))
    ddf = ddf.reset_index(drop=True)

    # Apply the dist_fix() function to each group separately
    group = ddf.groupby('trip')
    ddf = group.apply(dist_fix, meta=ddf._meta)
    ddf = ddf.reset_index(drop=True)

    ddf = ddf.map_partitions(normalize, meta=pd.DataFrame(columns=['exit_stop', 'target_stop', 'day_of_week', 'day_of_month', 'month','vehicle_id','distance', 'exit_time', 'arrive_time']))
    
    df = ddf.compute()
    mask = ~df['exit_stop'].isin(df['target_stop'])
    first_stops = set(df['exit_stop'][mask])

    routes = {elem: {elem} for elem in first_stops}

    ddf2 = dd.from_pandas(df, npartitions=20)

    group_v = ddf2.groupby('vehicle_id')
    ddf2 = group_v.apply(total_distances, routes, meta=pd.DataFrame(columns=['total_distance','first_stop','exit_stop', 'target_stop', 'day_of_week', 'day_of_month', 'month','vehicle_id' ,'distance', 'exit_time', 'arrive_time']))
    ddf2 = ddf2.reset_index(drop=True)

    ddf2 = ddf2.compute()

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
    
    ddf2.to_parquet(f'../processed_datasets/{date}.parquet', engine='pyarrow', schema=partition_schema)

    print(f"Finished processing {date} dataset.")

    return 1