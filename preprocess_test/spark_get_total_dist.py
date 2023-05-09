from pyspark.sql.functions import col, when, sum, broadcast
from pyspark.sql.types import *

def total_distance(df):
    df.cache()

    df = df.withColumn("total_distance", col("total_distance").cast("float"))

    # Get the unique stops
    unique_stops_df = df.select("first_stop").distinct()
    unique_stops_broadcast = broadcast(unique_stops_df.alias("s1"))

    for stop in df.select("first_stop").distinct().rdd.flatMap(lambda x: x).collect():
        stack = [(stop, 0)]
        processed_stops = set()
        path_distances = {}  # Dict to store distances for each path

        while stack:
            print(len(stack))

            curr_stop, curr_distance = stack.pop()

            if (df.filter(col("exit_stop") == curr_stop).agg(sum("distance")).collect()[0][0]) is not None:
                curr_distance += df.filter(col("exit_stop") == curr_stop).agg(sum("distance")).collect()[0][0]
            else:
                continue

            # Add the distance for the current path to the dict
            path_distances[(stop, curr_stop)] = curr_distance  
        
            df = df.withColumn("total_distance", when(col("exit_stop") == curr_stop, curr_distance).otherwise(col("total_distance")))

            target_stops = df.join(unique_stops_broadcast.alias("s2"), df.exit_stop == col("s2.first_stop"), "inner").select("target_stop").distinct().rdd.flatMap(lambda x: x).collect()

            for target_stop in target_stops:
                if target_stop not in processed_stops:
                    processed_stops.add(target_stop)
                    stack.append((target_stop, curr_distance))

        # Choose the minimum distance as the total_distance for this journey
        min_distance = min(path_distances.values())
        df = df.withColumn("total_distance", when(col("target_stop") == stop, min_distance).otherwise(col("total_distance")))

    df = df.filter(col("total_distance").isNotNull())
    return df

    