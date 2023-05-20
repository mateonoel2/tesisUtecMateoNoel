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
                .filter((col("label") >= 0.291) & (col("label") <= 0.458))
    
        # Save the DataFrame
        data.write.mode("overwrite").format("parquet").save("../processed_datasets_7_11/dataset")

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()