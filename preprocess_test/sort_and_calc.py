from datetime import datetime as dt, timedelta
import pandas as pd

def sort_and_calc(partition):
    
    #Clean unusefull data
    partition = partition.sort_values('time_received') \
                        .drop_duplicates(subset='distance_along_trip') \
                        .drop_duplicates(subset='time_received') \
                        .drop_duplicates(subset=['next_scheduled_stop_distance', 'next_scheduled_stop_id'])
    
    partition.reset_index(drop=True, inplace=True)
    
    data = pd.DataFrame(columns=['trip', 'distance', 'date', 'exit_time', 'arrive_time'])
    
    first = True
    skip_count = 0

    # iterate through each row in the dataframe
    for i in range(len(partition)-1):

        current = partition.iloc[[i]]
        next = partition.iloc[[i+1]]

        current_distance = current["distance_along_trip"].values[0]  
        next_distance = next["distance_along_trip"].values[0]
        
        next_stop_from_current= current["next_scheduled_stop_id"].values[0]
        next_stop_from_next = next["next_scheduled_stop_id"].values[0]

        current_time = dt.strptime(current["time_received"].values[0], '%Y-%m-%d %H:%M:%S')
        next_time = dt.strptime(next["time_received"].values[0], '%Y-%m-%d %H:%M:%S')   

        dist_next_stop_from_current = current["next_scheduled_stop_distance"].values[0]
        dist_next_stop_from_next = next["next_scheduled_stop_distance"].values[0]

        if next_distance<current_distance:
            first = True
            continue

        dist_two_times = next_distance - current_distance

        if(next_stop_from_current==next_stop_from_next):
            if (round(dist_two_times) != round(dist_next_stop_from_current - dist_next_stop_from_next)):
                first=True
                skip_count = 1
                continue
        
        if skip_count > 0:
            skip_count -= 1
            continue
    
        if next_stop_from_current!=next_stop_from_next:

            if (dist_next_stop_from_current > dist_two_times):
                first=True
                continue
            
            speed = dist_two_times / ((next_time - current_time).total_seconds())

            if speed<0.1 or speed>20:
                    first=True
                    continue
            
            # calculate arrival time and speed if this is the first stop on the route
            if first:
                prev_time_to_stop = current_time + timedelta(seconds=dist_next_stop_from_current/speed)
                prev_distance = dist_two_times - dist_next_stop_from_current
                prev_along_distance = next_distance 
                next_stop_from_prev = next_stop_from_current
                first = False

            else:

                #Calculate arrival time and speed if this is not the first stop on the route
                time_to_stop = dist_next_stop_from_current / speed
                arrived_time = current_time + timedelta(seconds=time_to_stop)
                    
                dist_two_stops = prev_distance + current_distance-prev_along_distance + dist_next_stop_from_current
            
                #Solo consideramos paradas entre 100 y 2000 metros.
                if(dist_two_stops<100 or dist_two_stops>2000):
                    first = True
                    continue
                
                new_row = [str(next_stop_from_prev)+str(next_stop_from_current), dist_two_stops, prev_time_to_stop.date(), prev_time_to_stop.time().replace(microsecond=0), arrived_time.time().replace(microsecond=0)]
                data.loc[len(data)] = new_row

                prev_time_to_stop = arrived_time
                prev_distance = dist_two_times - dist_next_stop_from_current
                prev_along_distance = next_distance 
                next_stop_from_prev = next_stop_from_current

    data.insert(0, 'vehicle_id', partition['vehicle_id'][0])
    return data