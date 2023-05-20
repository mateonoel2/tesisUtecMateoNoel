from pyspark.sql import SparkSession
from pyspark.sql.functions import col

if __name__ == '__main__':    
       spark = SparkSession.builder \
       .appName("parquet2 viewer") \
       .getOrCreate()

       data = spark.read.format("parquet").load("unNormalized_dataset/dataset")
       data.printSchema()

       data.show(1000)