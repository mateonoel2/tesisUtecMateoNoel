from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, round
from pyspark.sql.types import IntegerType

if __name__ == '__main__':    
       spark = SparkSession.builder \
       .appName("parquet2 viewer") \
       .getOrCreate()

       data = spark.read.format("parquet").load("processed_datasets_7_11/dataset")
       
       print(data.count())
       data.printSchema()

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
       df = df.withColumn("travel_time", (col("label") - col("exit_time")))
       df = df.withColumn("speed", col("distance") / col("travel_time"))
       df = df.withColumn("travel_time", (col("label") - col("first_time")))
       df = df.withColumn("total_speed", col("total_distance") / col("travel_time"))

       df = df.withColumn("speed", round("speed",2))
       df = df.withColumn("total_speed", round("total_speed",2))
       df = df.drop("travel_time")
       
       df = df.filter(col("speed") > 1)

       df.show(1000)

       df.write.mode("overwrite").format("parquet").save("unNormalized_dataset/dataset")
