from pyspark.sql import SparkSession
from pyspark.sql.functions import col

if __name__ == '__main__':    
    spark = SparkSession.builder \
    .appName("parquet2 viewer") \
    .getOrCreate()

    data = spark.read.format("parquet").load("processed_datasets2/dataset")\
           .filter(col("vehicle_id") == 371)\
           .limit(1000)

    data.show(1000)