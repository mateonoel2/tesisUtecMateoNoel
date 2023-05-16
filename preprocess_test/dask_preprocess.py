import dask.dataframe as dd
import os
from process_data import process_data
from dask.distributed import Client, wait
import dask.config
import time

dask.config.set({'distributed.comm.timeouts.connect': '2h'})

if __name__ == '__main__':    

    client = Client(timeout="2h", n_workers=70, threads_per_worker=1, memory_limit=None)
    print(client.scheduler_info()['address'])
    #client = Client('tcp://127.0.0.1:33779')
                    
    folder = "../datasets"
    datasets = os.listdir(folder)

    num_datasets = len(datasets)
    
    # Create a Dask dataframe from the CSV file
    for i in range(0, num_datasets): 
        dataset_name = os.path.join(folder, datasets[i]) 
        
        ddf = dd.read_csv(dataset_name, sep="\t", assume_missing=True,
                    usecols=['time_received', 'vehicle_id', 'distance_along_trip', 'inferred_phase',
                                'next_scheduled_stop_distance', 'next_scheduled_stop_id'])
            
        future = client.submit(process_data, ddf, priority=i)
        wait(future)

    print("SUCCESS")

    time.sleep(60*60*24)