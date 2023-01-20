#latitude	longitude	time_received	vehicle_id	distance_along_trip	inferred_direction_id	inferred_phase	inferred_route_id	inferred_trip_id	next_scheduled_stop_distance	next_scheduled_stop_id

#Se envía solo el tiempo cuando se llega a una parada
#BusId ParadaId TiempoAParada RecorridoDeParadaAnteriorAEsta 

file = open("../datasets/MTA-Bus-Time_.2014-08-01.txt", "r")
lines = file.readlines()

for i in range(1, len(lines)):
    list = lines[i].split("\t")
    BusId = list[3] 
   

