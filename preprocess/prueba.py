#BusId ParadaId TiempoDeLLegadaAParada RecorridoDeParadaAnteriorAEsta(m)
import os
import pandas as pd
from datetime import datetime as dt, timedelta

folder = "../datasets"
datasets = os.listdir(folder)

times = []
speeds = []

for dataset in datasets:
    df = pd.read_csv(folder+"/"+dataset, sep="\t")

    map_vehicles_by_id =  {}

    df.drop(df[df["inferred_phase"] == "LAYOVER_DURING"].index, inplace=True)

    for index, row in df.iterrows():
        currentVehicleId = row["vehicle_id"]
        if currentVehicleId != None and map_vehicles_by_id.get(currentVehicleId) == None:
            subDf = df[df["vehicle_id"] == currentVehicleId]
            map_vehicles_by_id[currentVehicleId] = subDf[["latitude", "longitude", "time_received", "distance_along_trip", "next_scheduled_stop_distance", "inferred_phase", "next_scheduled_stop_id"]]
            df.drop(subDf.index, inplace=True)

    df.dropna(subset=['vehicle_id', 'time_received', 'distance_along_trip'], inplace=True)
    # sort by time_received ascending df
    df.sort_values(by=['time_received'], inplace=True)

    vechicle_ids = df["vehicle_id"].unique()
    print(map_vehicles_by_id.keys())
    print(len(map_vehicles_by_id.keys()))
    print(len(vechicle_ids))

    for vehicle_id in map_vehicles_by_id.keys():
        first = True
        dist_purple = None
        is_in_progress = True

        for i in range(len(map_vehicles_by_id[vehicle_id])-1):
            current = map_vehicles_by_id[vehicle_id].iloc[[i]]
            current_pos = (current["latitude"].values[0], current["longitude"].values[0])
            current_distance = current["distance_along_trip"].values[0]
            next = map_vehicles_by_id[vehicle_id].iloc[[i+1]]
            next_pos = (next["latitude"].values[0], next["longitude"].values[0])
            next_distance = next["distance_along_trip"].values[0]
            if current_pos == next_pos or next_distance < current_distance:
                continue
            current_stop = current["next_scheduled_stop_id"].values[0]
            next_stop = next["next_scheduled_stop_id"].values[0]
            dist_two_times = next_distance - current_distance
            current_time = dt.strptime(current["time_received"].values[0], '%Y-%m-%d %H:%M:%S')
            next_time = dt.strptime(next["time_received"].values[0], '%Y-%m-%d %H:%M:%S')
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
                first = False
                if current_stop != next_stop:
                    # print(current_time, next_time, current_stop, next_stop)
                    speed = dist_two_times / (next_time - current_time).total_seconds()
                    prev_time_to_stop = current_time + timedelta(seconds=dist_current_next_stop/speed)
            else:
                if current_stop != next_stop:
                    speed = dist_two_times / (next_time - current_time).total_seconds()
                    if speed == 0:
                        dist_two_stops = dist_purple + next["next_scheduled_stop_distance"].values[0]
                        continue
                    time_to_stop = min(dist_two_times, dist_current_next_stop) / speed
                    arrived_time = current_time + timedelta(seconds=time_to_stop)
                    speed = dist_two_stops / (arrived_time-prev_time_to_stop).total_seconds()
                    prev_time_to_stop = arrived_time
                    dist_purple = None
                    times.append(arrived_time)
                    speeds.append(speed)

            if dist_purple == None:
                dist_purple = dist_two_times - dist_current_next_stop
                if dist_purple < 0:
                    dist_purple = 0
                dist_two_stops = dist_purple + next["next_scheduled_stop_distance"].values[0]
    break