from pyspark.sql import SparkSession
from pyspark.sql.functions import col

if __name__ == '__main__':    
    spark = SparkSession.builder \
    .appName("parquet viewer") \
    .getOrCreate()

    data = spark.read.format("parquet").load("processed_datasets/2014-09-10.parquet")
    data = data.drop("__null_dask_index__")
    data = data.orderBy(data.exit_time)
    data = data.drop("dat_of_week", "day_of_month", "month")
    data = data.withColumn("distance", col("distance") * 5000)
    data = data.withColumn("exit_time", col("exit_time") * 86400)
    data = data.withColumn("arrive_time", col("arrive_time") * 86400)

    data.show(10000)