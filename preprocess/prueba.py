#BusId ParadaId TiempoDeLLegadaAParada RecorridoDeParadaAnteriorAEsta(m)
import pandas as pd
from datetime import datetime as dt, timedelta

df = pd.read_csv("../datasets/MTA-Bus-Time_.2014-08-01.txt", sep="\t")

map_vehicles_by_id =  {}
max_vehicles = 1
cont_num_vehicles = 0

df.drop(df[df["inferred_phase"] == "LAYOVER_DURING"].index, inplace=True)

for index, row in df.iterrows():
    currentVehicleId = row["vehicle_id"]
    if currentVehicleId != None and map_vehicles_by_id.get(currentVehicleId) == None:
        subDf = df[df["vehicle_id"] == currentVehicleId]
        map_vehicles_by_id[currentVehicleId] = subDf[["time_received", "distance_along_trip", "next_scheduled_stop_distance", "next_scheduled_stop_id"]]
        df.drop(subDf.index, inplace=True)
        cont_num_vehicles += 1
    if cont_num_vehicles == max_vehicles:
        break

vehicle_id = 469
max_iters = 2
current_iters = 0
first = True
dist_purple = None
times = []
speeds = []
for i in range(len(map_vehicles_by_id[vehicle_id])):
    current = map_vehicles_by_id[vehicle_id].iloc[[i]]
    current_stop = current["next_scheduled_stop_id"].values[0]
    next = map_vehicles_by_id[vehicle_id].iloc[[i+1]]
    next_stop = next["next_scheduled_stop_id"].values[0]
    dist_two_times = next["distance_along_trip"].values[0] - current["distance_along_trip"].values[0]
    current_time = dt.strptime(current["time_received"].values[0], '%Y-%m-%d %H:%M:%S')
    next_time = dt.strptime(next["time_received"].values[0], '%Y-%m-%d %H:%M:%S')
    dist_current_next_stop = current["next_scheduled_stop_distance"].values[0]
    if first:
        first = False
        if current_stop != next_stop:
            speed = dist_two_times / (next_time - current_time).total_seconds()
            prev_time_to_stop = current_time + timedelta(seconds=dist_current_next_stop/speed)
    else:
        if current_stop != next_stop:
            speed = dist_two_times / (next_time - current_time).total_seconds()
            time_to_stop = dist_current_next_stop / speed
            arrived_time = current_time + timedelta(seconds=time_to_stop)
            times.append(arrived_time)
            speeds.append(dist_two_stops / (arrived_time-prev_time_to_stop).total_seconds())
            break
    # prev_next_distance = dist_current_next_stop
    if dist_purple == None:
        dist_purple = dist_two_times - dist_current_next_stop
        dist_two_stops = dist_purple + next["next_scheduled_stop_distance"].values[0]

print(times)
print(speeds)