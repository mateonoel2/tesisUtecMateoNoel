import pandas as pd
from datetime import datetime
import time

def time_to_seconds(time_obj):
    midnight = datetime.combine(datetime.today(), datetime.min.time())
    return (datetime.combine(datetime.today(), time_obj) - midnight).total_seconds()

def normalize(data):
    # Convert date feature to datetime format
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    # Extract additional date features
    data.insert(0, 'month', data['date'].dt.month)
    data.insert(0, 'day_of_month', data['date'].dt.day)
    data.insert(0, 'day_of_week', data['date'].dt.dayofweek)

    data = data.drop('date', axis=1)

    # Convert datetime to seconds since midnight
    data['exit_time'] = data['exit_time'].apply(time_to_seconds)
    data['arrive_time'] = data['arrive_time'].apply(time_to_seconds)

    # Divide 'exit_time' and 'arrive_time' columns by 86400 to normalize to [0, 1] range
    data['exit_time'] = data['exit_time'] / 86400
    data['arrive_time'] = data['arrive_time'] / 86400

    data.insert(0, 'target_stop', '')   
    data.insert(0, 'exit_stop', '')
     
    data[['exit_stop', 'target_stop']] = data['trip'].apply(lambda x: pd.Series([x.split('MTA_')[1], x.split('MTA_')[2]]))

    # convert to numeric data type
    data['exit_stop'] = data['exit_stop'].astype(int)
    data['target_stop'] = data['target_stop'].astype(int)
    data['first_stop'] = data['first_stop'].astype(int)

    data = data.drop('trip', axis=1)

    data['distance'] = data[['distance']] / 5000
    data['total_distance'] =  data[['total_distance']] / 5000
   
    return data