#BusId ParadaId TiempoDeLLegadaAParada RecorridoDeParadaAnteriorAEsta(m)
import pandas as pd

df = pd.read_csv("../datasets/MTA-Bus-Time_.2014-08-01.txt", sep="\t")

map_vehicles_by_id =  {}
max_vehicles = 10
cont_num_vehicles = 0

for index, row in df.iterrows():
    currentVehicleId = row["vehicle_id"]
    if currentVehicleId != None and map_vehicles_by_id.get(currentVehicleId) == None:
        subDf = df[df["vehicle_id"] == currentVehicleId]
        map_vehicles_by_id[currentVehicleId] = subDf["next_scheduled_stop_id", "next_scheduled_stop_distance"]
        df.drop(subDf.index, inplace=True)
        cont_num_vehicles += 1
    if cont_num_vehicles == max_vehicles:
        break

print(map_vehicles_by_id[469])