from datetime import datetime as dt, timedelta

def calc_data(partition):
    data = []

    first = True
    dist_purple = None
    is_in_progress = True
    prev_time_to_stop: dt | None = None

    for i in range(len(partition)-1):
        current = partition.iloc[[i]]
        current_pos = (current["latitude"].values[0], current["longitude"].values[0])
        current_distance = current["distance_along_trip"].values[0]
        next = partition.iloc[[i+1]]
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
                dist_purple = None

                if speed != 0:
                    data.append((current_stop, next_stop, dist_two_stops, prev_time_to_stop, arrived_time, speed))

                prev_time_to_stop = arrived_time

        if dist_purple == None:
            dist_purple = dist_two_times - dist_current_next_stop
            if dist_purple < 0:
                dist_purple = 0
            dist_two_stops = dist_purple + next["next_scheduled_stop_distance"].values[0]
    
    return data