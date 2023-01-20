#BusId ParadaId TiempoDeLLegadaAParada RecorridoDeParadaAnteriorAEsta(m)
import pandas as pd

df = pd.read_csv("../datasets/MTA-Bus-Time_.2014-08-01.txt", sep="\t")

for index, row in df.iterrows():
    currentVehicleId = row["vehicle_id"]
    subDf = df[df["vehicle_id"] == currentVehicleId]
    print(subDf["next_scheduled_stop_id"])
    break