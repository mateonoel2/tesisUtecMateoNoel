from pyspark.sql.functions import col, when, sum
from pyspark.sql.window import Window

def total_distance(df):
    df = df.withColumn("total_distance", col("total_distance").cast("float"))

    # Get the unique stops
    unique_stops = df.select("first_stop").distinct().rdd.flatMap(lambda x: x).collect()

    for stop in unique_stops:
        stack = [(stop, 0)]
        processed_stops = set()

        while stack:
            curr_stop, curr_distance = stack.pop()
            curr_distance += df.filter(col("exit_stop") == curr_stop).agg(sum("distance")).collect()[0][0]

            df = df.withColumn("total_distance", when(col("exit_stop") == curr_stop, curr_distance).otherwise(col("total_distance")))

            target_stops = df.filter(col("exit_stop") == curr_stop).select("target_stop").distinct().rdd.flatMap(lambda x: x).collect()
            
            for target_stop in target_stops:
                if target_stop not in processed_stops:
                    processed_stops.add(target_stop)
                    stack.append((target_stop, curr_distance))

    df = df.filter(col("total_distance").isNotNull())
    return df


