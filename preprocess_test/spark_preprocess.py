from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, min, max, lit, asc
from pyspark.sql.types import DoubleType
import sys

# Define a function to handle errors
def handle_error(e):
    print("Error: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        spark = SparkSession.builder \
        .appName("PySpark preprocess") \
        .config("spark.executor.instances", "1") \
        .config("spark.executor.cores", "80") \
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()

        # Load data from Parquet
        data = spark.read.parquet("../processed_data/dataset").coalesce(1).cache()

        # Get the unique values from exit_stop column
        exit_stops = data.select("exit_stop").distinct().rdd.flatMap(lambda x: x).collect()
        target_stops = data.select("target_stop").distinct().rdd.flatMap(lambda x: x).collect()

        all_stops = list(set(exit_stops + target_stops))

        #Get the unique values from vehicle_id column
        vehicle_ids = data.select("vehicle_id").distinct().rdd.flatMap(lambda x: x).collect()

        # Create a mapping from the old values to the new values (stops)
        mapping_s = dict(zip(all_stops, range(1, len(all_stops) + 1)))

        # Create UDFs to apply the mapping to each column
        exit_stop_mapping = udf(lambda exit_stop: mapping_s[exit_stop])
        target_stop_mapping = udf(lambda target_stop: mapping_s[target_stop])
        first_stop_mapping = udf(lambda first_stop: mapping_s[first_stop])

        # Create a mapping from the old values to the new values (vehicles)
        mapping_v = dict(zip(vehicle_ids, range(1, len(vehicle_ids) + 1)))

        # Create UDFs to apply the mapping to each column
        vehicle_mapping = udf(lambda vehicle_id: mapping_v[vehicle_id])
        
        # Define a UDF to normalize a column
        normalize_udf = udf(lambda x, min_val, max_val: (x - min_val) / (max_val - min_val), DoubleType())

        #Normalize distances
        agg_values = data.agg(min("total_distance"), max("total_distance")).collect()[0]
        min_value = agg_values[0]
        max_value = agg_values[1]

        #Print min and max values of total_distance
        print("Min and max values of total_distance:")
        print(min_value, max_value)

        # Apply the changes to the DataFrame using withColumn
        data = data \
            .withColumn("exit_stop", exit_stop_mapping(data["exit_stop"]).cast("integer")) \
            .withColumn("target_stop", target_stop_mapping(data["target_stop"]).cast("integer")) \
            .withColumn("first_stop", first_stop_mapping(data["first_stop"]).cast("integer"))\
            .withColumn("vehicle_id", vehicle_mapping(data["vehicle_id"]).cast("integer"))\
            .withColumn("distance", normalize_udf("distance", lit(min_value), lit(max_value)))\
            .withColumn("total_distance", normalize_udf("total_distance", lit(min_value), lit(max_value)))\
            .withColumnRenamed("arrive_time", "label")

        # Sort by time
        data = data.orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).coalesce(1)
        
        data.write.mode("overwrite").format("parquet").save("../processed_datasets2/dataset")

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()