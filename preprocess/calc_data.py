from datetime import datetime as dt, timedelta
import pandas as pd

def calc_data(partition):
    data = []

    first = True
    dist_purple = None
    is_in_progress = True
    prev_time_to_stop: dt | None = None # declare variable with type hinting

    # iterate through each row in the dataframe
    for i in range(len(partition)-1):
        current = partition.iloc[[i]]
        current_distance = current["distance_along_trip"].values[0]
        next = partition.iloc[[i+1]]
        next_distance = next["distance_along_trip"].values[0]
        current_time = dt.strptime(current["time_received"].values[0], '%Y-%m-%d %H:%M:%S')
        next_time = dt.strptime(next["time_received"].values[0], '%Y-%m-%d %H:%M:%S')

        # skip row if the current and next positions are the same, 
        # or if the next stop is not further along the route than the current stop,
        # or if the current and next times are the same
        if next_distance <= current_distance:
            continue
        
        while(current_time == next_time):
            i+=1
            partition = partition.drop(i)
            next = partition.iloc[[i+1]]
            next_distance = next["distance_along_trip"].values[0]
            next_time = dt.strptime(next["time_received"].values[0], '%Y-%m-%d %H:%M:%S')

        next_stop_from_current= current["next_scheduled_stop_id"].values[0]
        next_stop_from_next = next["next_scheduled_stop_id"].values[0]
        dist_next_stop_from_current = current["next_scheduled_stop_distance"].values[0]
        dist_next_stop_from_next = next["next_scheduled_stop_distance"].values[0]
        dist_two_times = next_distance - current_distance
        phase = current["inferred_phase"].values[0]

        if phase == "IN_PROGRESS":
            if is_in_progress == False:
                first = True
            is_in_progress = True
        else:
            is_in_progress = False
            continue

        if(dist_next_stop_from_next<=0):
            continue

        if (dist_two_times != dist_next_stop_from_current - dist_next_stop_from_next and next_stop_from_current==next_stop_from_next):
            next["next_scheduled_stop_distance"].values[0] = dist_next_stop_from_current - dist_two_times
         
       
        if next_stop_from_current!=next_stop_from_next:
            speed = dist_two_times / (next_time - current_time).total_seconds()
            if (speed<0.01 or speed>20):
                    continue
             # calculate arrival time and speed if this is the first stop on the route
            if first:
                prev_time_to_stop = current_time + timedelta(seconds=dist_next_stop_from_current/speed)
                first = False
            else:
                # calculate arrival time and speed if this is not the first stop on the route
                time_to_stop = dist_next_stop_from_current / speed
                arrived_time = current_time + timedelta(seconds=time_to_stop)

                dist_two_stops = dist_two_times-dist_next_stop_from_current + next["next_scheduled_stop_distance"].values[0]
                
                #Condición (SOLO SE ANALIZAN PARADAS DE MENOS DE 5KM)
                if(dist_two_stops>5000):
                    continue
                    
                # calculate the speed of the bus between the last two stops
                speed = dist_two_stops / (arrived_time-prev_time_to_stop).total_seconds()
                
                if speed != 0:
                    data.append((next_stop_from_current, next_stop_from_next, dist_two_stops, speed, prev_time_to_stop.weekday(),prev_time_to_stop.date(), prev_time_to_stop.time(), arrived_time.time()))

                prev_time_to_stop = arrived_time
    return data
