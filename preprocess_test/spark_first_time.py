from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import when, last, asc, col

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Load data from Parquet
df = spark.read.parquet("../processed_datasets2/dataset").coalesce(1).cache()

# Create a window specification to define the partition and ordering
window_spec = Window.partitionBy("first_stop").orderBy(asc("month"), asc("day_of_month"), asc("exit_time"))

# Add the "first_time" column
df = df.withColumn("first_time", when(df.first_stop == df.exit_stop, last(df.exit_time).over(window_spec))).coalesce(1).cache()

df = df.select("first_time", *df.columns[:-1])

#Fill null values in the "first_time" column with the last non-null value
df = df.withColumn("first_time", last(col("first_time"), ignorenulls=True).over(window_spec)).coalesce(1).cache()

df = df.dropna(subset=["first_time"]).coalesce(1).cache()

df = df.orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).coalesce(1).cache()

df.write.mode("overwrite").format("parquet").save("../processed_datasets3/dataset")