from pyspark.sql import SparkSession
from pyspark.sql.functions import col

if __name__ == '__main__':    
    spark = SparkSession.builder \
    .appName("parquet viewer") \
    .getOrCreate()

    data = spark.read.format("parquet").load("processed_datasets/*.parquet")

    data.show(1000)