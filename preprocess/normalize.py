import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import sys

def normalize(data):
    # Convert date feature to datetime format

    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    # Extract additional date features
    data.insert(0, 'month', data['date'].dt.month)
    data.insert(0, 'day_of_month', data['date'].dt.day)
    data.insert(0, 'day_of_week', data['date'].dt.dayofweek)

    data = data.drop('date', axis=1)

    # Define a function to convert time strings to seconds since midnight
    def time_to_seconds(time_str):
        time_str = time_str.strftime('%H:%M:%S')
        dt = datetime.strptime(time_str, '%H:%M:%S')
        midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return (dt - midnight).total_seconds()

    # Iterate over the 'exit_time' and 'arrive_time' columns and convert each value to seconds since midnight
    for col in ['exit_time', 'arrive_time']:
        data[col] = data[col].apply(time_to_seconds)

    data.insert(0, 'target_stop', '')   
    data.insert(0, 'exit_stop', '')
     
    data[['exit_stop', 'target_stop']] = data['trip'].apply(lambda x: pd.Series(['MTA_' + x.split('MTA_')[1], 'MTA_' + x.split('MTA_')[2]]))

    data = data.drop('trip', axis=1)

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

    # Divide 'exit_time' and 'arrive_time' columns by 86400 to normalize to [0, 1] range
    data['exit_time'] = data['exit_time'] / 86400
    data['arrive_time'] = data['arrive_time'] / 86400
    
    return data