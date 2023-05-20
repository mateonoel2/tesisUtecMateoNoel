from pyspark.sql import SparkSession
from pyspark.sql.functions import asc, col
import sys

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        # Create a SparkSession
        spark = SparkSession.builder \
        .appName("Concat Long and short distances")\
        .config("spark.executor.instances", "10")\
        .config("spark.executor.cores", "8")\
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()
        
        print("Solo datos que salen desde las 7am y llegan hasta las 11am")

        # Load data from Parquet
        data = spark.read.parquet("../processed_datasets3/dataset").coalesce(1)\
                .filter((col("exit_time") >= 0.291) & (col("label") <= 0.458)).coalesce(1)
        
        data1 = data.select("month", "day_of_month", "day_of_week", "exit_time", "exit_stop", "distance", "target_stop", "label").cache()

        data2 = data.select("month", "day_of_month", "day_of_week", "first_time", "first_stop", "total_distance", "target_stop", "label").cache()

        data2 = data2.withColumnRenamed("first_stop", "exit_stop") \
             .withColumnRenamed("total_distance", "distance") \
             .withColumnRenamed("first_time", "exit_time") \
             .cache()

        # Concatenate the DataFrames
        data = data1.unionAll(data2)

        # Remove duplicates
        data = data.dropDuplicates().cache().coalesce(1)

        data = data.orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).cache()

        data = data.drop("month", "day_of_month").cache()

        data.show(1000)

        # Save the DataFrame
        data.write.mode("overwrite").format("parquet").save("../processed_datasets_concat/dataset")

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()