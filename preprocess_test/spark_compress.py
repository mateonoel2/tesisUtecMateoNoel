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
        .appName("PySpark compress") \
        .config("spark.executor.instances", "10") \
        .config("spark.executor.cores", "8") \
        .getOrCreate()

        path = "../processed_datasets/*.parquet"

        # Load data from Parquet
        data = spark.read.format("parquet").load(path)
        data = data.coalesce(1)
        data.write.mode("overwrite").format("parquet").save("../processed_data/dataset")

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()