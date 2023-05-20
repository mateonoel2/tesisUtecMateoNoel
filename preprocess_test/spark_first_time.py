from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import when, last, asc, col, lag

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Load data from Parquet
df = spark.read.parquet("../processed_datasets2/dataset").coalesce(1).cache()

# Create a window specification to define the partition and ordering
window_spec = Window.partitionBy("vehicle_id").orderBy(asc("month"), asc("day_of_month"), asc("exit_time"))

# Add the "first_time" column
df = df.withColumn("first_time", when(col("first_stop") != lag(col("first_stop")).over(window_spec),col("exit_time")).otherwise(None))
df = df.withColumn("window_values", when(col("first_stop") != lag(col("first_stop")).over(window_spec),col("exit_stop")).otherwise(None))

#Fill null values in the "first_time" column with the last non-null value
df = df.withColumn("first_time", last(col("first_time"), ignorenulls=True).over(window_spec))
df = df.withColumn("window_values", last(col("window_values"), ignorenulls=True).over(window_spec))

df = df.filter(col("first_stop") == col("window_values"))

df = df.drop("window_values")

df = df.dropna(subset=["first_time"])

df = df.select("first_time", *df.columns[:-1])

df = df.orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).coalesce(1).cache()

df.write.mode("overwrite").format("parquet").save("../processed_datasets3/dataset")