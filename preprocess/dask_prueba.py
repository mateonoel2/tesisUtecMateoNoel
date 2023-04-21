import os
import threading
import dask.dataframe as dd
from datetime import datetime as dt, timedelta

folder = "../datasets"
datasets = os.listdir(folder)

def calc_data(sub_df, lock):
    start_stops_id = []
    end_stops_id = []
    start_times = []
    distances = []
    speeds = []
    arrived_times = []

    first = True
    dist_purple = None
    is_in_progress = True
    prev_time_to_stop: dt = None

    for i in range(len(sub_df)-1):
        lock.acquire()
        current = sub_df.iloc[[i]]
        current_pos = (current["latitude"].values[0], current["longitude"].values[0])
        current_distance = current["distance_along_trip"].values[0]
        next = sub_df.iloc[[i+1]]
        next_pos = (next["latitude"].values[0], next["longitude"].values[0])
        next_distance = next["distance_along_trip"].values[0]
        current_time = dt.strptime(current["time_received"].values[0], '%Y-%m-%d %H:%M:%S')
        next_time = dt.strptime(next["time_received"].values[0], '%Y-%m-%d %H:%M:%S')
        if current_pos == next_pos or next_distance <= current_distance or current_time == next_time:
            continue
        current_stop = current["next_scheduled_stop_id"].values[0]
        next_stop = next["next_scheduled_stop_id"].values[0]
        dist_two_times = next_distance - current_distance
        dist_current_next_stop = current["next_scheduled_stop_distance"].values[0]
        phase = current["inferred_phase"].values[0]

        if phase == "IN_PROGRESS":
            if is_in_progress == False:
                first = True
                dist_purple = None
            is_in_progress = True
        else:
            is_in_progress = False
            continue

        if first:
            if current_stop != next_stop:
                speed = dist_two_times / (next_time - current_time).total_seconds()
                prev_time_to_stop = current_time + timedelta(seconds=dist_current_next_stop/speed)
                first = False
        else:
            if current_stop != next_stop:
                speed = dist_two_times / (next_time - current_time).total_seconds()
                if speed == 0:
                    dist_two_stops = dist_purple + next["next_scheduled_stop_distance"].values[0]
                    continue
                time_to_stop = min(dist_two_times, dist_current_next_stop) / speed
                arrived_time = current_time + timedelta(seconds=time_to_stop)
                if dist_two_stops == 0 or prev_time_to_stop >= arrived_time:
                    dist_two_stops = dist_purple + next["next_scheduled_stop_distance"].values[0]
                    continue
                speed = dist_two_stops / (arrived_time-prev_time_to_stop).total_seconds()
                prev_time_to_stop = arrived_time
                dist_purple = None
                if speed != 0:
                    start_stops_id.append(current_stop)
                    end_stops_id.append(next_stop)
                    distances.append(dist_two_stops)
                    speeds.append(speed)
                    arrived_times.append(arrived_time)


        if dist_purple == None:
            dist_purple = dist_two_times - dist_current_next_stop
            if dist_purple < 0:
                dist_purple = 0
            dist_two_stops = dist_purple + next["next_scheduled_stop_distance"].values[0]
        lock.release()
    
    print(len(arrived_times), len(speeds))
    return (start_stops_id, end_stops_id, distances, arrived_times, speeds)

lock = threading.Lock()

for dataset in datasets:
    df = dd.read_csv(os.path.join(folder, dataset), sep="\t", assume_missing=True)
    df = df.loc[~(df['inferred_phase'] == 'LAYOVER_DURING')]
    df = df.loc[(df['vehicle_id'] == 469.0) | (df['vehicle_id'] == 195.0)]
    df = df.dropna(subset=['vehicle_id', 'time_received', 'next_scheduled_stop_distance', 'distance_along_trip'])
    grouped = df.groupby('vehicle_id')
    result = grouped.apply(calc_data, lock, meta=('float64'))
    aux = result.compute()
    # write this into a csv
    aux.to_csv("results/result.csv")
    break