import dask.dataframe as dd
import os
from calc_data import calc_data
from normalize import normalize
import pandas as pd
from dask.distributed import Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Define a function to sort the partition by 'time_received'
def sort_and_calc(df):
    sorted_df = df.sort_values('time_received')
    return calc_data(sorted_df)

def dist_fix(df):
    df.loc[:,'distance'] = df['distance'].round(decimals=-1)
    mode = df['distance'].mode()[0]
    df = df[(df['distance'] > mode - 20) & (df['distance'] < mode + 20)]
    df.loc[:, 'distance'] = mode
    return df

def normalize2(data):
    # Concatenate the exit_stop and target_stop columns
    stops = pd.concat([data['exit_stop'], data['target_stop']], ignore_index=True)

    # Fit the encoder on the concatenated stops and transform the columns
    le = LabelEncoder()
    stops_encoded = le.fit_transform(stops)
    data['exit_stop'] = stops_encoded[:len(data)]
    data['target_stop'] = stops_encoded[len(data):]

    # Scale numerical variables
    scaler = MinMaxScaler()
    data['distance'] = scaler.fit_transform(data[['distance']])
    return data

if __name__ == '__main__':
    
    client = Client()

    folder = "../datasets"
    datasets = os.listdir(folder)

    # Load your Dask DataFrame
    dataset_names = []
    for dataset in datasets: 
        dataset_names.append(os.path.join(folder, dataset))

    ddf = dd.read_csv(dataset_names, sep="\t", assume_missing=True, usecols=['time_received', 'vehicle_id', 'distance_along_trip', 'inferred_phase', 'next_scheduled_stop_distance', 'next_scheduled_stop_id'])

    # Filter out all phases that aren't LAYOVER_DURING and all rows with null
    ddf = ddf.loc[(ddf['inferred_phase'] != "LAYOVER_DURING")].dropna()

    #test line
    #ddf = ddf.loc[(ddf['vehicle_id'] == 469.0) | (ddf['vehicle_id'] == 195.0)]

    # Apply the sort_and_calc() function to each group separately
    group = ddf.groupby('vehicle_id')
    ddf = group.apply(sort_and_calc, meta=pd.DataFrame(columns=['trip', 'distance', 'date', 'exit_time', 'arrive_time']))
 
    # Apply the dist_fix() function to each group separately
    group = ddf.groupby('trip')
    ddf = group.apply(dist_fix, meta=ddf._meta)
 
    ddf = ddf.reset_index(drop=True)

    ddf = ddf.map_partitions(normalize, meta=pd.DataFrame(columns=['exit_stop', 'target_stop', 'day_of_week', 'day_of_month', 'month','distance', 'exit_time', 'arrive_time']))
    
    ddf.visualize()
    
    # Drop all groups and reset index
    pdf = ddf.compute()

    pdf = normalize2(pdf)

    # save the Pandas dataframe in Parquet format
    pdf.to_parquet('../processed_datasets/test.parquet', engine='pyarrow')