import pandas as pd
from datetime import datetime

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

    data = data.drop('trip', axis=1)
    
    mask = ~data['exit_stop'].isin(data['target_stop'])
    first_stops = set(data['exit_stop'][mask])

    data.insert(0, 'first_stop', None)
    data.insert(0, 'total_distance', 0.0)

    routes = {elem: {(elem, 0.0)} for elem in first_stops}
    loop_count = 0

    #longest_route_limit = 100 stops
    while data['first_stop'].isna().any() and loop_count < 100:
        for route_name, route_dic in routes.items():
            stops = [stop[0] for stop in route_dic]
            t_dists = {stop[0]: stop[1] for stop in route_dic}

            mask = data['exit_stop'].isin(stops) & data['first_stop'].isna()

            data.loc[mask,'total_distance'] = data.loc[mask,'distance'] + data.loc[mask, 'exit_stop'].map(t_dists)

            new_stops = data.loc[mask, 'target_stop'].to_list()
            new_dists = data.loc[mask, 'total_distance'].to_list()
            new_elems = set(zip(new_stops, new_dists))

            routes[route_name] = routes[route_name] | new_elems
            data.loc[mask,'first_stop'] = route_name
            
        loop_count += 1

    
    if loop_count == 100:
        data = data.dropna(subset=['first_stop'])

    return data