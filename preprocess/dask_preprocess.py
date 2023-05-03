import dask.dataframe as dd
import os
from process_data import process_data
from dask.distributed import Client, wait
import dask.bag as db

if __name__ == '__main__':    

    client = Client()

    folder = "../datasets"
    datasets = os.listdir(folder)
    dataset_names = []

    for dataset in datasets:
        dataset_names.append(os.path.join(folder, dataset))
    
    # use a wait condition to wait for an available client
    while not client:
        client = Client()
        wait([client])

    # Create a Dask bag of Dask dataframes
    dataframes_bag = db.from_sequence(dataset_names).map(lambda fn: dd.read_csv(fn, sep="\t", assume_missing=True, usecols=['time_received', 'vehicle_id', 'distance_along_trip', 'inferred_phase', 'next_scheduled_stop_distance', 'next_scheduled_stop_id']))
    
    # use a wait condition to wait for an available client
    while not client:
        client = Client()
        wait([client])

    dataframes_bag.map(process_data).compute()

    client.close()