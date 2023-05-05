from pyspark.sql import SparkSession

if __name__ == '__main__':    
    spark = SparkSession.builder \
    .appName("text viewer") \
    .getOrCreate()

    data = spark.read.format("csv").load("datasets/MTA-Bus-Time_.2014-08-01.txt", sep='\t')
    data = data.drop("_c0" , "_c1", "_c5", "_c6", "_c7","_c8")

    data.show(100000)