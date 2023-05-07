import dask.dataframe as dd
import os
from process_data import process_data
from dask.distributed import Client
import dask.config

dask.config.set({'distributed.comm.timeouts.connect': '2h'})

if __name__ == '__main__':    

    client = Client(timeout="2h", n_workers=15, threads_per_worker=89)

    folder = "../datasets"
    datasets = os.listdir(folder)

    num_datasets = len(datasets)
    counter = 0
    
    for i in range(counter, num_datasets):
        dataset_name = os.path.join(folder, datasets[i])

        # Create a Dask dataframe from the CSV file
        ddf = dd.read_csv(dataset_name, sep="\t", assume_missing=True, usecols=['time_received', 'vehicle_id', 'distance_along_trip', 'inferred_phase', 'next_scheduled_stop_distance', 'next_scheduled_stop_id'])

        # submit the computation to the cluster
        future = client.submit(process_data, ddf)

    #wait for the results  
    result = future.result()

    print("SUCCESS")
    client.close()