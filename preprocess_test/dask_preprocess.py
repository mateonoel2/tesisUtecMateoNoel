import dask.dataframe as dd
import os
from process_data import process_data
from dask.distributed import Client, wait
import dask.config

dask.config.set({'distributed.comm.timeouts.connect': '2h'})

if __name__ == '__main__':    

    client = Client(timeout="2h", n_workers=80, threads_per_worker=1, memory_limit=None)

    folder = "../datasets/"
    datasets = os.path.join(folder, "*.txt") 
    
    ddf = dd.read_csv(datasets, sep="\t", assume_missing=True,
                usecols=['time_received', 'vehicle_id', 'distance_along_trip', 'inferred_phase',
                            'next_scheduled_stop_distance', 'next_scheduled_stop_id'])
        
    future = client.submit(process_data, ddf)
        
    wait(future)
    print("SUCCESS")