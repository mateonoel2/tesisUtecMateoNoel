from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, round
from pyspark.sql.types import IntegerType

if __name__ == '__main__':    
       spark = SparkSession.builder \
       .appName("parquet2 viewer") \
       .getOrCreate()

       data = spark.read.format("parquet").load("processed_datasets3/dataset")\
              .filter(col("vehicle_id") == 1)\
              .limit(1000)
       
       # dictionary containing the minimum and maximum values for each column
       max_dist = 41260
       min_max_dict = {'total_distance': (100, max_dist), 'distance': (100, max_dist), 'first_time': (0, 86400), 'exit_time': (0, 86400), 'label': (0, 86400)}

       # create a new dataframe with unnormalized columns
       df = data.select(    [ col_name for col_name in data.columns if col_name not in min_max_dict.keys()] +
                            [ when(col(col_name).isNull(), col(col_name))
                              .otherwise(col(col_name) * (max_val - min_val) + min_val)
                              .alias(col_name)
                              for col_name, (min_val, max_val) in min_max_dict.items()] 
                     )

       df = df.withColumn('exit_time', round('exit_time').cast(IntegerType()))\
              .withColumn('label', round('label').cast(IntegerType()))\
              .withColumn('distance', round('distance').cast(IntegerType()))\
              .withColumn('first_time', round('first_time').cast(IntegerType()))\
              .withColumn('total_distance', round('total_distance').cast(IntegerType()))

       # show the unnormalized dataframe
       df = df.withColumn('distance(km)', col('distance') / 1000)
       df = df.withColumn("travel_time", ((col("label") - col("exit_time")) / 3600))
       df = df.withColumn("speed(km/h)", col("distance(km)") / col("travel_time"))

       df = df.withColumn("speed(km/h)", round("speed(km/h)",2))
       df = df.drop("travel_time", "distance(km)")
       df = df.drop("vehicle_id", "month", "day_of_month")

       df.show(1000)

       # df.select(col("vehicle_id"), col("month"), col("day_of_month").alias("day"), 
       #           col("distance").alias("distance(m)"), col("exit_time").alias("exit_time(s)"), 
       #           col("label").alias("arrive_time(s)"), col("speed(km/h)")).show(20)

       # df.select(col("vehicle_id"),col("first_stop"), col("exit_stop"), 
       #           col("target_stop"), col("total_distance").alias("total_distance(m)"), 
       #           col("distance").alias("distance(m)")).show(20)