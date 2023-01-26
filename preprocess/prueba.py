#BusId ParadaId TiempoDeLLegadaAParada RecorridoDeParadaAnteriorAEsta(m)
import pandas as pd

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

print(map_vehicles_by_id[469])