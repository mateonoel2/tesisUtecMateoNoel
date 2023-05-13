from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, min, max, lit, asc
from pyspark.sql.types import DoubleType
from datetime import datetime, timedelta
import os
import sys

# Define a function to handle errors
def handle_error(e):
    print("Error: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        spark = SparkSession.builder \
        .appName("PySpark preprocess") \
        .config("spark.driver.memory", "20g") \
        .config("spark.executor.memory", "90g") \
        .config("spark.executor.instances", "80") \
        .config("spark.executor.cores", "1") \
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()


        path = "../processed_datasets/dataset.parquet"

        # Load data from Parquet
        data = spark.read.format("parquet").load(path)
        data = data.coalesce(1)
        data = data.dropna().dropDuplicates().coalesce(80)

        # Get the unique values from exit_stop column
        exit_stops = data.select("exit_stop").distinct().rdd.flatMap(lambda x: x).collect()
        target_stops = data.select("target_stop").distinct().rdd.flatMap(lambda x: x).collect()

        all_stops = list(set(exit_stops + target_stops))

        # Create a mapping from the old values to the new values
        mapping = dict(zip(all_stops, range(1, len(all_stops) + 1)))

        # Create UDFs to apply the mapping to each column
        exit_stop_mapping = udf(lambda exit_stop: mapping[exit_stop])
        target_stop_mapping = udf(lambda target_stop: mapping[target_stop])
        first_stop_mapping = udf(lambda first_stop: mapping[first_stop])

        # Apply the mapping to the DataFrame using withColumn
        data = data \
            .withColumn("exit_stop", exit_stop_mapping(data["exit_stop"]).cast("integer")) \
            .withColumn("target_stop", target_stop_mapping(data["target_stop"]).cast("integer")) \
            .withColumn("first_stop", first_stop_mapping(data["first_stop"]).cast("integer"))
        
        # Rename arrive column
        data = data.withColumnRenamed("arrive_time", "label")

        #Normalize distances
        min_value = data.agg(min("distance")).collect()[0][0]
        max_value = data.agg(max("total_distance")).collect()[0][0]
        
        # Define a UDF to normalize a column
        normalize_udf = udf(lambda x, min_val, max_val: (x - min_val) / (max_val - min_val), DoubleType())

        data = data.withColumn("distance", normalize_udf("distance", lit(min_value), lit(max_value)))
        data = data.withColumn("total_distance", normalize_udf("total_distance", lit(min_value), lit(max_value)))

        #Ordenar por tiempo
        data = data.orderBy(asc("month"), asc("day_of_month"), asc("exit_time"))

        data = data.coalesce(1)

        data.write.mode("overwrite").format("parquet").save("../processed_datasets2/dataset")

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()