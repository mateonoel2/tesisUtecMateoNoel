from dist_fix import dist_fix
from sort_and_calc import sort_and_calc
from normalize import normalize
import pandas as pd
import pyarrow as pa
from dask.distributed import get_client
import dask.dataframe as dd
import time

# def total_distance(df):
#     df['total_distance'] = df['total_distance'].astype(float)
#     unique_stops = df['first_stop'].unique()
    
#     for stop in unique_stops:
#         stack = [(stop, 0)]
#         processed_stops = set()
#         while stack:
#             curr_stop, curr_distance = stack.pop()
#             curr_distance += df.loc[df['exit_stop'] == curr_stop, 'distance']
#             df.loc[df['exit_stop'] == curr_stop, 'total_distance'] =  curr_distance
#             target_stops = df.loc[df['exit_stop'] == curr_stop, 'target_stop'].tolist()
#             for target_stop in target_stops:
#                 if target_stop not in processed_stops:
#                     processed_stops.add(target_stop)
#                     stack.append((target_stop, curr_distance))

#     df = df.dropna()
#     return df

def get_first_stops(df):
    mask = ~df['exit_stop'].isin(df['target_stop'])
    first_stops = df['exit_stop'][mask]
    return first_stops.drop_duplicates()

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

    #probar con dos vehicle id
    ddf = ddf.loc[(ddf['vehicle_id'] == 469.0)]

    # Apply the sort_and_calc() function to each group separately
    group = ddf.groupby('vehicle_id')
    ddf = group.apply(sort_and_calc, meta=pd.DataFrame(columns=['trip', 'distance', 'date', 'exit_time', 'arrive_time']))
    ddf = ddf.reset_index(drop=True)

    # Apply the dist_fix() function to each group separately
    group = ddf.groupby('trip')
    ddf = group.apply(dist_fix, meta=ddf._meta)
    ddf = ddf.reset_index(drop=True)

    ddf = ddf.map_partitions(normalize, meta=pd.DataFrame(columns=['exit_stop', 'target_stop', 'day_of_week', 'day_of_month', 'month','distance', 'exit_time', 'arrive_time']))
    
    #find starting stopse
    first_stops = ddf.map_partitions(get_first_stops)
    first_stops = first_stops.compute()
    
    ddf = ddf.compute()

    routes = {elem: {elem} for elem in first_stops}
    ddf.insert(0, 'total_distance', 0)
    ddf.insert(0, 'first_stop', None)
  
    while ddf['first_stop'].isna().any():
        for route_name, stops in routes.items():
            mask = ddf['exit_stop'].isin(stops)
            new_stops = ddf.loc[mask, 'target_stop'].to_list()
            routes[route_name] = routes[route_name] | set(new_stops)
            ddf.loc[mask, 'first_stop'] = ddf.loc[mask, 'first_stop'].fillna(route_name)
            #ddf.loc[mask, 'total_distance'] + ddf.loc[mask, 'first_stop'].fillna(ddf.loc[mask, 'total_distance'] + ddf.loc[mask, 'distance'])

    ddf['target_stop'] = ddf['target_stop'].astype(int)
    
    #ddf.insert(0, 'total_distance', None)
  
    # ddf2 = dd.from_pandas(ddf, npartitions=10)
    
    # # Apply the total_distance function to each group separately
    # group = ddf2.groupby('first_stop')

    # ddf2 = group.apply(total_distance, meta=ddf2._meta)
  
    # ddf2 = ddf2.reset_index(drop=True)


    partition_schema = pa.schema([
        pa.field('total_distance', pa.float64()),
        pa.field('first_stop', pa.int64()),
        pa.field('exit_stop', pa.int64()),
        pa.field('target_stop', pa.int64()),
        pa.field('day_of_week', pa.int32()),
        pa.field('day_of_month', pa.int32()),
        pa.field('month', pa.int32()),
        pa.field('distance', pa.float64()),
        pa.field('exit_time', pa.float64()),
        pa.field('arrive_time', pa.float64()),
    ])
    

    # print(ddf.head)
    # time.sleep(100)
    # Write processed data to file
    ddf.to_parquet(f'../processed_datasets/{date}.parquet', engine='pyarrow', schema=partition_schema)

    print(f"Finished processing {date} dataset.")

    return 1