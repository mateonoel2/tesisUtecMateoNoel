import dask.dataframe as dd
import os
from process_data import process_data
from dask.distributed import Client, wait
import dask.config
import time

dask.config.set({'distributed.comm.timeouts.connect': '2h'})

if __name__ == '__main__':    

    client = Client(timeout="2h", n_workers=80, threads_per_worker=1, memory_limit=None)

    folder = "../datasets"
    datasets = os.listdir(folder)

    num_datasets = len(datasets)
    results = []
    
    # Create a Dask dataframe from the CSV file
    for i in range(0, num_datasets): 
        dataset_name = os.path.join(folder, datasets[i]) 
        
        ddf = dd.read_csv(dataset_name, sep="\t", assume_missing=True,
                    usecols=['time_received', 'vehicle_id', 'distance_along_trip', 'inferred_phase',
                                'next_scheduled_stop_distance', 'next_scheduled_stop_id'])
            
        future = client.submit(process_data, ddf, priority=num_datasets-i-1)
        results.append(future)
        
    wait(results)
    print("SUCCESS")